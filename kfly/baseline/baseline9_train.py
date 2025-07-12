# baseline8_train.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # For feature selection
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import json
import os
import joblib # For saving/loading scaler and label_encoders
import gc # For garbage collection

# --- 1. 配置和路径定义 ---
script_dir = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(script_dir, '..', 'data', 'train_data', 'train.csv')
OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'baseline8_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'is_new_did'
RANDOM_SEED = 42
BATCH_SIZE = 2048 # 根据服务器配置，可以尝试更大，但需要测试显存
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
L2_REG_WEIGHT = 1e-5
EARLY_STOPPING_PATIENCE = 7 # 稍微增加耐心，给模型更多机会
F1_THRESHOLD_PREDICTION = 0.5 # 用于F1-score的预测阈值

UNSEEN_CATEGORY_PLACEHOLDER = '___UNSEEN___'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for training: {device}")

# --- 2. 辅助函数：处理 udmap 字段 ---
def parse_udmap_helper(udmap_str):
    if isinstance(udmap_str, str) and udmap_str != '{}':
        try:
            data = json.loads(udmap_str)
            return data.get('botId', None), data.get('pluginId', None)
        except json.JSONDecodeError:
            return None, None
    return None, None

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

    df[['botId', 'pluginId']] = df['udmap'].apply(lambda x: pd.Series(parse_udmap_helper(x)))
    df['botId'] = df['botId'].fillna('unknown_bot').astype(str)
    df['pluginId'] = df['pluginId'].fillna('unknown_plugin').astype(str)
    df = df.drop(columns=['udmap'])

    for col in categorical_id_cols + ['botId', 'pluginId']:
        df[col] = df[col].astype(str)

    df['did'] = df['did'].astype(str) # Retain did for aggregation and sequential features
    return df

# --- 4. 特征工程 ---
def feature_engineer(df):
    print("Starting feature engineering...")
    
    # 转换为 datetime 对象
    df['common_ts_dt'] = pd.to_datetime(df['common_ts'], unit='ms')

    # 对数据进行排序，为序列特征做准备
    df = df.sort_values(by=['did', 'common_ts_dt']).reset_index(drop=True)

    # 1. 时间戳特征
    df['hour'] = df['common_ts_dt'].dt.hour.astype(str)
    df['dayofweek'] = df['common_ts_dt'].dt.dayofweek.astype(str)
    df['month'] = df['common_ts_dt'].dt.month.astype(str)
    df['year'] = df['common_ts_dt'].dt.year.astype(str)

    # 2. 序列和相对时间特征 (来自 LightGBM Baseline)
    print("  Calculating sequential and relative time features...")
    df['event_rank'] = df.groupby('did').cumcount() + 1
    
    did_first_event_info = df.groupby('did').agg(
        common_ts_min=('common_ts_dt', 'first'),
        first_eid=('eid', 'first')
    ).reset_index()
    
    df = df.merge(did_first_event_info, on='did', how='left')
    df['time_since_first'] = (df['common_ts_dt'] - df['common_ts_min']).dt.total_seconds().fillna(0) # 填充可能因为单事件用户导致的NaN

    # 3. DID 聚合特征
    print("  Calculating DID-based aggregated features...")
    user_agg_df = df.groupby('did').agg(
        user_total_events=('mid', 'size'),
        user_unique_mids=('mid', 'nunique'),
        user_unique_eids=('eid', 'nunique'),
        user_unique_device_brands=('device_brand', 'nunique'),
        user_avg_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').mean()),
        user_max_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').max()),
        user_min_appver=('appver', lambda x: pd.to_numeric(x, errors='coerce').min()),
        ts_span=('common_ts_dt', lambda x: (x.max() - x.min()).total_seconds()) # 用户活跃时长
    ).reset_index()
    
    df = df.merge(user_agg_df, on='did', how='left')

    # 处理 first_eid 的填充和类型 (与 LightGBM 一致)
    df['first_eid'] = df['first_eid'].fillna(-1).astype(str) # 转换为字符串，LabelEncoder会处理

    # 删除原始时间戳列和datetime对象列，以及did和first_eid的原始列（如果需要）
    df = df.drop(columns=['common_ts', 'common_ts_dt', 'common_ts_min', 'did']) 
    print("Feature engineering completed.")
    return df

# --- 5. 特征编码与标准化 ---
def preprocess_features(df, numerical_features, categorical_features, scaler=None, label_encoders=None, is_train_mode=True):
    print("Starting feature preprocessing (encoding and scaling)...")

    if is_train_mode:
        scaler = StandardScaler()
        label_encoders = {}
        embedding_dims_map = {}
    else:
        embedding_dims_map = {}

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
        embed_dim = min(50, max(2, num_categories // 2))
        embedding_dims_map[col] = (num_categories, embed_dim)
        encoded_categorical_data[col] = df[f'{col}_encoded'].values

    for col in numerical_features:
        if df[col].isnull().any():
            if is_train_mode:
                fill_value = df[col].mean()
            else:
                # Use mean from training set if available in scaler
                if scaler is not None and hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
                    col_index = np.where(scaler.feature_names_in_ == col)[0][0]
                    fill_value = scaler.mean_[col_index]
                else:
                    fill_value = df[col].mean() # Fallback, should not happen if scaler is fitted correctly
            df[col] = df[col].fillna(fill_value)

    if is_train_mode:
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        scaler.feature_names_in_ = np.array(numerical_features) # Store feature names for consistency
    else:
        df[numerical_features] = scaler.transform(df[numerical_features])

    print("Feature preprocessing completed.")
    return df, scaler, label_encoders, embedding_dims_map, encoded_categorical_data

# --- 6. 特征选择 ---
def select_features(df, numerical_features, categorical_features, target_col):
    print("Starting feature selection using RandomForestClassifier...")
    features_for_selection = numerical_features + [f'{col}_encoded' for col in categorical_features]
    X_for_selection = df[features_for_selection].values
    y_for_selection = df[target_col].values

    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_for_selection, y_for_selection)

    feature_importances = pd.Series(rf_model.feature_importances_, index=features_for_selection)
    feature_importances = feature_importances.sort_values(ascending=False)

    print("Feature Importances (Top 20):")
    print(feature_importances.head(20))

    importance_threshold = 0.001 # 降低阈值以包含更多特征，这些新特征可能很重要
    selected_features_encoded = feature_importances[feature_importances > importance_threshold].index.tolist()

    final_numerical_features = [f for f in selected_features_encoded if f in numerical_features]
    final_categorical_features = [f.replace('_encoded', '') for f in selected_features_encoded if f.endswith('_encoded')]

    print(f"\nSelected {len(final_numerical_features) + len(final_categorical_features)} features for modeling.")
    print(f"Final Numerical Features: {final_numerical_features}")
    print(f"Final Categorical Features: {final_categorical_features}")

    return final_numerical_features, final_categorical_features

# --- 7. PyTorch Dataset 和 DataLoader ---
class CustomUserDataset(TensorDataset):
    def __init__(self, numerical_data, categorical_data_dict, labels):
        all_tensors = [numerical_data] + list(categorical_data_dict.values()) + [labels]
        super(CustomUserDataset, self).__init__(*all_tensors)
        self.categorical_keys = list(categorical_data_dict.keys())
        self._num_numerical_features = numerical_data.shape[1]

    def __getitem__(self, idx):
        item_tensors = super(CustomUserDataset, self).__getitem__(idx)
        numerical_feats = item_tensors[0]
        categorical_feats_item_dict = {}
        for i, key in enumerate(self.categorical_keys):
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

        self.fc1 = nn.Linear(self.total_input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.output_layer = nn.Linear(64, 1)

    def forward(self, numerical_feats, categorical_feats_dict):
        embedded_feats = []
        for col, embed_layer in self.embeddings.items():
            if col in categorical_feats_dict:
                input_tensor = categorical_feats_dict[col]
                embedded_feats.append(embed_layer(input_tensor))

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

        if self.mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf

    def __call__(self, current_score, model):
        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_model_state = model.state_dict()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else: # 'min'
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_model_state = model.state_dict()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

# --- 主执行流程 ---
if __name__ == '__main__':
    print("--- Starting Full Data Training Pipeline ---")

    # 1. 加载和清洗数据 (完整数据)
    df_full_train = load_and_clean_data(TRAIN_DATA_PATH)

    # 2. 特征工程 (包含 DID 聚合特征和序列特征)
    df_full_train = feature_engineer(df_full_train)

    # 定义所有可能的数值和类别特征 (在特征工程后获取)
    # Target列 'is_new_did' 是唯一非特征列，'did' 列已被删除
    all_numerical_features = [col for col in df_full_train.columns if df_full_train[col].dtype in ['int64', 'float64'] and col != TARGET]
    all_categorical_features = [col for col in df_full_train.columns if df_full_train[col].dtype == 'object' ]

    # 3. 预处理 (编码和标准化) - 第一次运行，用于特征选择
    # 注意：这里对整个 df_full_train 进行预处理，方便后续特征选择
    df_processed_for_selection, scaler_for_selection, label_encoders_for_selection, embedding_dims_map_for_selection, _ = \
        preprocess_features(df_full_train.copy(), all_numerical_features, all_categorical_features, is_train_mode=True)

    # 4. 特征选择
    final_numerical_features, final_categorical_features = \
        select_features(df_processed_for_selection, all_numerical_features, all_categorical_features, TARGET)
    
    del df_processed_for_selection, scaler_for_selection, label_encoders_for_selection, embedding_dims_map_for_selection
    gc.collect() # 强制垃圾回收

    # 5. 重新准备数据，只包含选择的特征，并进行训练集/验证集划分
    X_full_features = df_full_train.drop(columns=[TARGET])
    y_full = df_full_train[TARGET]
    del df_full_train # Free df_full_train memory
    gc.collect()

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_full_features, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
    )
    del X_full_features, y_full
    gc.collect()

    # 对训练集进行预处理（fit_transform），使用选中的最终特征
    X_train_processed, scaler_final, label_encoders_final, embedding_dims_map_final_train, encoded_cat_train_data = \
        preprocess_features(X_train_df.copy(), final_numerical_features, final_categorical_features, is_train_mode=True)
    del X_train_df
    gc.collect()

    # 对验证集进行预处理（transform），使用训练集fit的scaler和encoders
    X_val_processed, _, _, _, encoded_cat_val_data = \
        preprocess_features(X_val_df.copy(), final_numerical_features, final_categorical_features,
                            scaler=scaler_final, label_encoders=label_encoders_final, is_train_mode=False)
    del X_val_df
    gc.collect()
    
    # 转换为 PyTorch 张量
    X_train_num_tensor = torch.tensor(X_train_processed[final_numerical_features].values, dtype=torch.float32)
    X_val_num_tensor = torch.tensor(X_val_processed[final_numerical_features].values, dtype=torch.float32)

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


    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 6. 实例化模型、损失函数、优化器
    model = NewUserPredictor(
        num_numerical_features=len(final_numerical_features),
        embedding_dims={col: embedding_dims_map_final_train[col] for col in final_categorical_features}
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) # <-- 加入 pos_weight
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
        
        all_preds_binary = (np.array(all_preds_prob) > F1_THRESHOLD_PREDICTION).astype(int)
        current_f1_score = f1_score(all_labels, all_preds_binary)
        auc_score = roc_auc_score(all_labels, all_preds_prob)
        accuracy = accuracy_score(all_labels, all_preds_binary)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val F1: {current_f1_score:.4f}, Val AUC: {auc_score:.4f}, Val Acc: {accuracy:.4f}")

        # 早停判断
        if early_stopper(current_f1_score, model):
            print(f"Early stopping triggered at epoch {epoch+1}!")
            model.load_state_dict(early_stopper.best_model_state)
            print(f"Best Val F1-score: {early_stopper.best_score:.4f}")
            break
        
        # 保存最佳模型
        if current_f1_score == early_stopper.best_score and early_stopper.counter == 0: # Only save if this is the new best
            model_save_path = os.path.join(OUTPUT_DIR, f'best_model_f1_{current_f1_score:.4f}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    print("\n--- Training Finished ---")

    # 保存最终使用的 scaler 和 label_encoders 以及最终特征列表和 embedding_dims_map_final_train
    joblib.dump(scaler_final, os.path.join(OUTPUT_DIR, 'scaler_final.joblib'))
    joblib.dump(label_encoders_final, os.path.join(OUTPUT_DIR, 'label_encoders_final.joblib'))
    
    # 保存特征列表和 embedding_dims_map，predict时需要
    feature_config = {
        'final_numerical_features': final_numerical_features,
        'final_categorical_features': final_categorical_features,
        'embedding_dims_map': embedding_dims_map_final_train,
        'f1_threshold_prediction': F1_THRESHOLD_PREDICTION # 保存预测阈值
    }
    joblib.dump(feature_config, os.path.join(OUTPUT_DIR, 'feature_config.joblib'))
    
    print(f"Scaler, LabelEncoders, and Feature Config saved to {OUTPUT_DIR}")

    print("\n--- End of Training Script ---")