import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_curve
import json
import os
import joblib
import gc
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. 配置和路径定义 ---
script_dir = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(script_dir, '..', 'data', 'train_data', 'train.csv')
OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'baseline10_output') # 更改输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'is_new_did'
RANDOM_SEED = 42
BATCH_SIZE = 2048
NUM_EPOCHS = 5000 # 算力充足可调大，有早停保护
LEARNING_RATE = 0.001
L2_REG_WEIGHT = 1e-4 # 增大L2正则化权重
EARLY_STOPPING_PATIENCE = 10 # 稍微增加耐心
# F1_THRESHOLD_PREDICTION = 0.5 # 这个将动态确定

UNSEEN_CATEGORY_PLACEHOLDER = '___UNSEEN___'
UNSEEN_UDMAP_KEY_PLACEHOLDER = 'UNSEEN_KEY' # 新增用于udmap键的placeholder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for training: {device}")

# --- 2. 辅助函数：处理 udmap 字段 ---
def parse_udmap_extended_helper(udmap_str):
    parsed_data = {'botId': 'unknown_bot', 'pluginId': 'unknown_plugin', 'udmap_keys_str': ''}
    if isinstance(udmap_str, str) and udmap_str != '{}':
        try:
            data = json.loads(udmap_str)
            # 提取 botId 和 pluginId
            parsed_data['botId'] = str(data.get('botId', 'unknown_bot'))
            parsed_data['pluginId'] = str(data.get('pluginId', 'unknown_plugin'))
            
            # 提取所有键并用空格连接，用于文本特征处理
            parsed_data['udmap_keys_str'] = ' '.join(sorted(data.keys()))
            
            # 也可以在这里提取其他通用字段，例如：
            # parsed_data['custom_key_X'] = str(data.get('custom_key_X', ''))
            
        except json.JSONDecodeError:
            pass # 保持默认值或已有的 None
    return parsed_data

# --- 3. 数据加载与基础清洗 ---
def load_and_clean_data(file_path):
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    numerical_cols = ['common_ts']
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_id_cols = [
        'mid', 'eid', 'device_brand', 'ntt', 'operator',
        'common_country', 'common_province', 'common_city',
        'appver', 'channel', 'os_type'
    ]
    for col in categorical_id_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- udmap 新的处理方式 ---
    udmap_parsed_series = df['udmap'].apply(parse_udmap_extended_helper)
    df_udmap_features = pd.json_normalize(udmap_parsed_series) # 展平字典到新的列

    df = pd.concat([df.drop(columns=['udmap']), df_udmap_features], axis=1) # 替换原始udmap列
    
    # 将udmap提取的特征也添加到类别特征中
    categorical_id_cols.extend(['botId', 'pluginId']) # 这两个已经通过udmap_parsed_series提取出来
    
    for col in categorical_id_cols: # 确保所有这些列都是字符串类型
        if col in df.columns: # 确保列存在
            df[col] = df[col].fillna(f'unknown_{col}').astype(str) # 填充为unknown_col

    df['did'] = df['did'].astype(str)
    
    return df, categorical_id_cols # 返回更新后的 categorical_id_cols

# --- 4. 特征工程 ---
def feature_engineer(df):
    print("Starting feature engineering...")
    
    df['common_ts_dt'] = pd.to_datetime(df['common_ts'], unit='ms')
    df = df.sort_values(by=['did', 'common_ts_dt']).reset_index(drop=True)

    df['hour'] = df['common_ts_dt'].dt.hour.astype(str)
    df['dayofweek'] = df['common_ts_dt'].dt.dayofweek.astype(str)
    df['month'] = df['common_ts_dt'].dt.month.astype(str)
    df['year'] = df['common_ts_dt'].dt.year.astype(str)

    print("  Calculating sequential and relative time features...")
    df['event_rank'] = df.groupby('did').cumcount() + 1
    
    did_first_event_info = df.groupby('did').agg(
        common_ts_min=('common_ts_dt', 'first'),
        first_eid=('eid', 'first')
    ).reset_index()
    
    df = df.merge(did_first_event_info, on='did', how='left')
    df['time_since_first'] = (df['common_ts_dt'] - df['common_ts_min']).dt.total_seconds().fillna(0)

    print("  Calculating DID-based aggregated features...")
    user_agg_df = df.groupby('did').agg(
        user_total_events=('mid', 'size'),
        user_unique_mids=('mid', 'nunique'),
        user_unique_eids=('eid', 'nunique'),
        user_unique_device_brands=('device_brand', 'nunique'),
        user_avg_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').mean()),
        user_max_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').max()),
        user_min_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').min()),
        ts_span=('common_ts_dt', lambda x: (x.max() - x.min()).total_seconds())
    ).reset_index()
    
    df = df.merge(user_agg_df, on='did', how='left')

    df['first_eid'] = df['first_eid'].fillna(-1).astype(str)

    df = df.drop(columns=['common_ts', 'common_ts_dt', 'common_ts_min', 'did']) 
    print("Feature engineering completed.")
    return df

# --- 5. 特征编码与标准化 ---
def preprocess_features(df, numerical_features, categorical_features, text_features,
                        scaler=None, label_encoders=None, vectorizers=None, is_train_mode=True):
    print("Starting feature preprocessing (encoding and scaling)...")

    if is_train_mode:
        scaler = StandardScaler()
        label_encoders = {}
        vectorizers = {}
        embedding_dims_map = {}
    else:
        embedding_dims_map = {} # 在测试模式下不需要重新计算

    encoded_categorical_data = {}
    for col in categorical_features:
        if is_train_mode:
            le = LabelEncoder()
            unique_vals = df[col].astype(str).unique().tolist()
            if UNSEEN_CATEGORY_PLACEHOLDER not in unique_vals:
                unique_vals.append(UNSEEN_CATEGORY_PLACEHOLDER)
            
            le.fit(unique_vals)
            df[f'{col}_encoded'] = le.transform(df[col].astype(str))
            label_encoders[col] = le
        else: # Evaluation mode (transform only)
            unseen_mask = ~df[col].astype(str).isin(label_encoders[col].classes_)
            df.loc[unseen_mask, col] = UNSEEN_CATEGORY_PLACEHOLDER
            df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
            df[f'{col}_encoded'] = df[f'{col}_encoded'].astype(int)

        num_categories = len(label_encoders[col].classes_)
        embed_dim = min(100, max(2, num_categories // 4)) # 增大嵌入维度
        embedding_dims_map[col] = (num_categories, embed_dim)
        encoded_categorical_data[col] = df[f'{col}_encoded'].values

    # --- 处理文本特征 ---
    text_numerical_features_names = []
    for col in text_features:
        if is_train_mode:
            vectorizer = CountVectorizer(max_features=50) # 限制特征数量，避免维度爆炸
            # 确保文本列没有NaN，并填充UNSEEN_UDMAP_KEY_PLACEHOLDER
            df[col] = df[col].fillna(UNSEEN_UDMAP_KEY_PLACEHOLDER)
            text_matrix = vectorizer.fit_transform(df[col])
            vectorizers[col] = vectorizer
        else:
            # 填充测试集中的未知值，确保它们在词汇表中存在或映射到未知
            unseen_text_mask = ~df[col].astype(str).isin(vectorizers[col].get_feature_names_out())
            df.loc[unseen_text_mask, col] = UNSEEN_UDMAP_KEY_PLACEHOLDER
            df[col] = df[col].fillna(UNSEEN_UDMAP_KEY_PLACEHOLDER) # 再次填充以防万一
            text_matrix = vectorizers[col].transform(df[col])
        
        # 将稀疏矩阵转换为DataFrame，并添加到df中
        text_df = pd.DataFrame(text_matrix.toarray(), columns=[f'{col}_text_{i}' for i in range(text_matrix.shape[1])], index=df.index)
        df = pd.concat([df, text_df], axis=1)
        text_numerical_features_names.extend(text_df.columns.tolist())

    # --- 数值特征标准化 ---
    all_numerical_features_to_scale = numerical_features + text_numerical_features_names
    for col in all_numerical_features_to_scale:
        if df[col].isnull().any():
            if is_train_mode:
                fill_value = df[col].mean()
            else:
                if scaler is not None and hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
                    col_index = np.where(scaler.feature_names_in_ == col)[0][0]
                    fill_value = scaler.mean_[col_index]
                else: # Fallback should not happen if scaler is fitted correctly
                    fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)

    if is_train_mode:
        df[all_numerical_features_to_scale] = scaler.fit_transform(df[all_numerical_features_to_scale])
        scaler.feature_names_in_ = np.array(all_numerical_features_to_scale) # Store feature names for consistency
    else:
        df[all_numerical_features_to_scale] = scaler.transform(df[all_numerical_features_to_scale])

    print("Feature preprocessing completed.")
    return df, scaler, label_encoders, vectorizers, embedding_dims_map, encoded_categorical_data, text_numerical_features_names

# --- 6. 特征选择 ---
def select_features(df, numerical_features, categorical_features, text_numerical_features_names, target_col):
    print("Starting feature selection using RandomForestClassifier...")
    
    # 结合所有待选择的特征名称
    features_for_selection = numerical_features + text_numerical_features_names + [f'{col}_encoded' for col in categorical_features]
    
    # 确保所有特征列都存在
    features_for_selection = [f for f in features_for_selection if f in df.columns]

    X_for_selection = df[features_for_selection].values
    y_for_selection = df[target_col].values

    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_for_selection, y_for_selection)

    feature_importances = pd.Series(rf_model.feature_importances_, index=features_for_selection)
    feature_importances = feature_importances.sort_values(ascending=False)

    print("Feature Importances (Top 20):")
    print(feature_importances.head(20))

    importance_threshold = 0.0005 # 进一步降低阈值，以捕获更多可能与udmap相关的特征
    selected_features_encoded = feature_importances[feature_importances > importance_threshold].index.tolist()

    final_numerical_features = [f for f in selected_features_encoded if f in numerical_features or f in text_numerical_features_names]
    final_categorical_features = [f.replace('_encoded', '') for f in selected_features_encoded if f.endswith('_encoded')]

    print(f"\nSelected {len(final_numerical_features) + len(final_categorical_features)} features for modeling.")
    print(f"Final Numerical Features: {final_numerical_features}")
    print(f"Final Categorical Features: {final_categorical_features}")

    return final_numerical_features, final_categorical_features

# --- 7. PyTorch Dataset 和 DataLoader ---
class CustomUserDataset(TensorDataset):
    def __init__(self, numerical_data, categorical_data_dict, labels):
        if numerical_data.dim() == 1:
            numerical_data = numerical_data.unsqueeze(1)
        
        # 确保分类特征的键按照排序后的顺序存储，以便在__getitem__中正确取回
        self.categorical_keys_ordered = sorted(categorical_data_dict.keys())
        all_categorical_tensors_list = [categorical_data_dict[key] for key in self.categorical_keys_ordered]

        all_tensors = [numerical_data] + all_categorical_tensors_list + [labels]
        super(CustomUserDataset, self).__init__(*all_tensors)
        
        self._num_numerical_features = numerical_data.shape[1]

    def __getitem__(self, idx):
        item_tensors = super(CustomUserDataset, self).__getitem__(idx)
        numerical_feats = item_tensors[0]
        categorical_feats_item_dict = {}
        
        # 按照存储时的顺序取出分类特征
        for i, key in enumerate(self.categorical_keys_ordered):
            categorical_feats_item_dict[key] = item_tensors[1 + i]
        
        label = item_tensors[-1]
        return numerical_feats, categorical_feats_item_dict, label

# --- 8. 深度学习模型定义 ---
class NewUserPredictor(nn.Module):
    def __init__(self, num_numerical_features, embedding_dims):
        super(NewUserPredictor, self).__init__()

        self.embeddings = nn.ModuleDict()
        embedding_output_dim = 0
        for col, (num_categories, embed_dim) in embedding_dims.items():
            self.embeddings[col] = nn.Embedding(num_categories, embed_dim)
            embedding_output_dim += embed_dim

        self.total_input_dim = num_numerical_features + embedding_output_dim

        # 增加网络宽度
        self.fc1 = nn.Linear(self.total_input_dim, 512) # 增大
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4) # 增大Dropout

        self.fc2 = nn.Linear(512, 256) # 增大
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4) # 增大Dropout

        self.fc3 = nn.Linear(256, 128) # 增大
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3) # 增大Dropout

        self.output_layer = nn.Linear(128, 1) # 对应fc3输出

    def forward(self, numerical_feats, categorical_feats_dict):
        embedded_feats = []
        # 确保遍历顺序与 CustomUserDataset 中的 key 顺序一致
        for col in sorted(categorical_feats_dict.keys()):
            if col in self.embeddings:
                input_tensor = categorical_feats_dict[col]
                embedded_feats.append(self.embeddings[col](input_tensor))

        if embedded_feats:
            combined_feats = torch.cat(embedded_feats + [numerical_feats], dim=1)
        else:
            combined_feats = numerical_feats

        x = F.relu(self.bn1(self.fc1(combined_feats)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        return self.output_layer(x)

# --- 9. 早停机制 (Early Stopping Callback) ---
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_f1_threshold = 0.5 # 保存最佳F1对应的阈值

        if self.mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf

    def __call__(self, current_score, model, current_f1_threshold=0.5): # 增加阈值参数
        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_model_state = model.state_dict()
                self.best_f1_threshold = current_f1_threshold # 保存最佳阈值
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else: # 'min'
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_model_state = model.state_dict()
                self.best_f1_threshold = current_f1_threshold # 保存最佳阈值
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

# --- 主执行流程 ---
if __name__ == '__main__':
    print("--- Starting Full Data Training Pipeline ---")

    # 1. 加载和清洗数据 (完整数据)
    df_full_train, all_categorical_features_after_udmap = load_and_clean_data(TRAIN_DATA_PATH)

    # 2. 特征工程 (包含 DID 聚合特征和序列特征)
    df_full_train = feature_engineer(df_full_train)

    # 定义所有可能的数值和类别特征 (在特征工程后获取)
    # Target列 'is_new_did' 是唯一非特征列，'did' 列已被删除
    all_numerical_features_raw = [col for col in df_full_train.columns if df_full_train[col].dtype in ['int64', 'float64'] and col != TARGET]
    # 更新 categorical_features 列表，确保包含所有 udmap 相关的类别特征
    # all_categorical_features 在 load_and_clean_data 中已更新，这里再添加 hour, dayofweek, month, year
    additional_cat_features = ['hour', 'dayofweek', 'month', 'year']
    all_categorical_features = list(set(all_categorical_features_after_udmap + additional_cat_features))
    
    # 定义文本特征列
    text_features = ['udmap_keys_str'] # udmap_keys_str 是由 udmap 提取出的文本特征

    # 3. 预处理 (编码和标准化) - 第一次运行，用于特征选择
    df_processed_for_selection, scaler_for_selection, label_encoders_for_selection, vectorizers_for_selection, \
    embedding_dims_map_for_selection, _, text_numerical_features_names_for_selection = \
        preprocess_features(df_full_train.copy(), all_numerical_features_raw, all_categorical_features, text_features, is_train_mode=True)

    # 4. 特征选择
    final_numerical_features, final_categorical_features = \
        select_features(df_processed_for_selection, all_numerical_features_raw, all_categorical_features, 
                        text_numerical_features_names_for_selection, TARGET)
    
    del df_processed_for_selection, scaler_for_selection, label_encoders_for_selection, vectorizers_for_selection, embedding_dims_map_for_selection
    del text_numerical_features_names_for_selection
    gc.collect()

    # 5. 重新准备数据，只包含选择的特征，并进行训练集/验证集划分
    X_full_features = df_full_train.drop(columns=[TARGET])
    y_full = df_full_train[TARGET]
    del df_full_train
    gc.collect()

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_full_features, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
    )
    del X_full_features, y_full
    gc.collect()

    # 对训练集进行预处理（fit_transform），使用选中的最终特征
    X_train_processed, scaler_final, label_encoders_final, vectorizers_final, embedding_dims_map_final_train, \
    encoded_cat_train_data, final_text_numerical_features_names = \
        preprocess_features(X_train_df.copy(), final_numerical_features, final_categorical_features, text_features, is_train_mode=True)
    del X_train_df
    gc.collect()

    # 对验证集进行预处理（transform），使用训练集fit的scaler和encoders
    X_val_processed, _, _, _, _, encoded_cat_val_data, _ = \
        preprocess_features(X_val_df.copy(), final_numerical_features, final_categorical_features, text_features,
                            scaler=scaler_final, label_encoders=label_encoders_final, vectorizers=vectorizers_final, is_train_mode=False)
    del X_val_df
    gc.collect()
    
    # 转换为 PyTorch 张量
    # 确保只选择 final_numerical_features 中属于数值和文本处理后数值的列
    all_final_numerical_cols_for_tensor = [col for col in final_numerical_features if col in X_train_processed.columns]
    X_train_num_tensor = torch.tensor(X_train_processed[all_final_numerical_cols_for_tensor].values, dtype=torch.float32)
    X_val_num_tensor = torch.tensor(X_val_processed[all_final_numerical_cols_for_tensor].values, dtype=torch.float32)


    categorical_train_tensors = {
        col: torch.tensor(encoded_cat_train_data[col], dtype=torch.long)
        for col in final_categorical_features
    }
    categorical_val_tensors = {
        col: torch.tensor(encoded_cat_val_data[col], dtype=torch.long)
        for col in final_categorical_features
    }
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    del X_train_processed, X_val_processed, encoded_cat_train_data, encoded_cat_val_data, y_train, y_val
    gc.collect()

    # 计算 pos_weight 处理不平衡数据
    pos_count = y_train_tensor.sum()
    neg_count = len(y_train_tensor) - pos_count
    pos_weight = neg_count / pos_count
    print(f"Positive samples: {int(pos_count)}, Negative samples: {int(neg_count)}, Pos_weight: {pos_weight.item():.2f}")

    # --- 创建 CustomUserDataset 实例 ---
    train_dataset = CustomUserDataset(X_train_num_tensor, categorical_train_tensors, y_train_tensor)
    val_dataset = CustomUserDataset(X_val_num_tensor, categorical_val_tensors, y_val_tensor)
    # --- 实例创建结束 ---

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 6. 实例化模型、损失函数、优化器
    model = NewUserPredictor(
        num_numerical_features=len(all_final_numerical_cols_for_tensor), # 使用实际传入的数值特征数量
        embedding_dims={col: embedding_dims_map_final_train[col] for col in final_categorical_features}
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG_WEIGHT)

    # 7. 初始化早停
    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=0.0001, mode='max')
    
    print("\n--- Starting Model Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (numerical_data, categorical_data_dict_batch, labels) in enumerate(train_loader):
            numerical_data = numerical_data.to(device)
            labels = labels.to(device)
            categorical_data_dict_batch_on_device = {
                k: v.to(device) for k, v in categorical_data_dict_batch.items()
            }

            optimizer.zero_grad()
            outputs = model(numerical_data, categorical_data_dict_batch_on_device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Current Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0
        all_preds_prob = []
        all_labels = []
        with torch.no_grad():
            for numerical_data, categorical_data_dict_batch, labels in val_loader:
                numerical_data = numerical_data.to(device)
                labels = labels.to(device)
                categorical_data_dict_batch_on_device = {
                    k: v.to(device) for k, v in categorical_data_dict_batch.items()
                }
                outputs = model(numerical_data, categorical_data_dict_batch_on_device)
                val_loss += criterion(outputs, labels).item()
                all_preds_prob.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        all_preds_prob = np.array(all_preds_prob).flatten()
        all_labels = np.array(all_labels).flatten()

        # --- 动态寻找最佳 F1 阈值 ---
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds_prob)
        best_f1 = 0
        best_threshold_epoch = 0.5
        for i in range(len(thresholds)):
            if precision[i] + recall[i] == 0: # 避免除零
                continue
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_epoch = thresholds[i]
        # --- 动态寻找结束 ---

        auc_score = roc_auc_score(all_labels, all_preds_prob)
        # 使用最佳阈值计算当前 F1 和准确率
        all_preds_binary = (all_preds_prob > best_threshold_epoch).astype(int)
        accuracy = accuracy_score(all_labels, all_preds_binary)


        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val F1 (Best Thresh): {best_f1:.4f} (Thresh: {best_threshold_epoch:.4f}), "
              f"Val AUC: {auc_score:.4f}, Val Acc: {accuracy:.4f}")

        # 早停判断
        if early_stopper(best_f1, model, best_f1_threshold): # 将最佳F1和对应阈值传入早停器
            print(f"Early stopping triggered at epoch {epoch+1}!")
            model.load_state_dict(early_stopper.best_model_state)
            print(f"Best Val F1-score: {early_stopper.best_score:.4f} (at threshold: {early_stopper.best_f1_threshold:.4f})")
            break
        
        # 保存最佳模型
        if best_f1 == early_stopper.best_score and early_stopper.counter == 0: # 只有当是新的最佳F1时才保存
            model_save_path = os.path.join(OUTPUT_DIR, f'best_model_f1_{best_f1:.4f}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    print("\n--- Training Finished ---")

    # 保存最终使用的 scaler 和 label_encoders 以及最终特征列表和 embedding_dims_map_final_train
    joblib.dump(scaler_final, os.path.join(OUTPUT_DIR, 'scaler_final.joblib'))
    joblib.dump(label_encoders_final, os.path.join(OUTPUT_DIR, 'label_encoders_final.joblib'))
    joblib.dump(vectorizers_final, os.path.join(OUTPUT_DIR, 'vectorizers_final.joblib')) # 保存文本向量化器
    
    # 保存特征列表和 embedding_dims_map，predict时需要
    feature_config = {
        'final_numerical_features': all_final_numerical_cols_for_tensor, # 保存精确的数值特征列表
        'final_categorical_features': final_categorical_features,
        'embedding_dims_map': embedding_dims_map_final_train,
        'final_text_numerical_features_names': final_text_numerical_features_names, # 保存文本转化的数值特征名
        'f1_threshold_prediction': early_stopper.best_f1_threshold # 保存训练时找到的最佳F1阈值
    }
    joblib.dump(feature_config, os.path.join(OUTPUT_DIR, 'feature_config.joblib'))
    
    print(f"Scaler, LabelEncoders, Vectorizers, and Feature Config saved to {OUTPUT_DIR}")

    print("\n--- End of Training Script ---")