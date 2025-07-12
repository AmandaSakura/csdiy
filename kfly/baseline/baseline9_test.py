# baseline8_predict.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import json
import os
import joblib # For loading scaler and label_encoders

# --- 1. 配置和路径定义 ---
script_dir = os.path.dirname(__file__)

TEST_DATA_PATH = os.path.join(script_dir, '..', 'data', 'test_data', 'testA_data.csv')
OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'baseline9_output') # 从这里加载模型和预处理器
SUBMISSION_FILE_PATH = os.path.join(OUTPUT_DIR, 'submission_baseline8.csv')

UNSEEN_CATEGORY_PLACEHOLDER = '___UNSEEN___'

# --- 2. 共享辅助函数 (与训练代码保持一致) ---
def parse_udmap_helper(udmap_str):
    if isinstance(udmap_str, str) and udmap_str != '{}':
        try:
            data = json.loads(udmap_str)
            return data.get('botId', None), data.get('pluginId', None)
        except json.JSONDecodeError:
            return None, None
    return None, None

# --- 3. 数据加载与基础清洗 (与训练代码保持一致) ---
def load_and_clean_data_for_predict(file_path): # Renamed to avoid conflict if you import
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # 填充数值缺失值
    numerical_cols = ['common_ts']
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 填充类别缺失值
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

    df['did'] = df['did'].astype(str)
    return df

# --- 4. 特征工程 (与训练代码保持一致，尤其是DID聚合逻辑) ---
def feature_engineer_for_predict(df): # Renamed
    print("Starting feature engineering for test data...")
    
    # 转换为 datetime 对象
    df['common_ts_dt'] = pd.to_datetime(df['common_ts'], unit='ms')

    # 对数据进行排序，为序列特征做准备
    df = df.sort_values(by=['did', 'common_ts_dt']).reset_index(drop=True)

    # 1. 时间戳特征
    df['hour'] = df['common_ts_dt'].dt.hour.astype(str)
    df['dayofweek'] = df['common_ts_dt'].dt.dayofweek.astype(str)
    df['month'] = df['common_ts_dt'].dt.month.astype(str)
    df['year'] = df['common_ts_dt'].dt.year.astype(str)

    # 2. 序列和相对时间特征
    print("  Calculating sequential and relative time features...")
    df['event_rank'] = df.groupby('did').cumcount() + 1
    
    did_first_event_info = df.groupby('did').agg(
        common_ts_min=('common_ts_dt', 'first'),
        first_eid=('eid', 'first')
    ).reset_index()
    
    df = df.merge(did_first_event_info, on='did', how='left')
    df['time_since_first'] = (df['common_ts_dt'] - df['common_ts_min']).dt.total_seconds().fillna(0)

    # 3. DID 聚合特征
    print("  Calculating DID-based aggregated features for test data...")
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

    # 处理 first_eid 的填充和类型
    df['first_eid'] = df['first_eid'].fillna(-1).astype(str)
    
    # original_dids will be managed in main block of predict script
    df = df.drop(columns=['common_ts', 'common_ts_dt', 'common_ts_min', 'did']) 
    print("Feature engineering completed for test data.")
    return df

# --- 5. 特征编码与标准化 (与训练代码共享，但is_train_mode=False) ---
def preprocess_features_for_predict(df, numerical_features, categorical_features, scaler, label_encoders, is_train_mode=False): # Renamed
    print("Starting feature preprocessing (encoding and scaling) for test data...")

    encoded_categorical_data = {}
    for col in categorical_features:
        unseen_mask = ~df[col].astype(str).isin(label_encoders[col].classes_)
        df.loc[unseen_mask, col] = UNSEEN_CATEGORY_PLACEHOLDER
        df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
        df[f'{col}_encoded'] = df[f'{col}_encoded'].astype(int)

        encoded_categorical_data[col] = df[f'{col}_encoded'].values

    for col in numerical_features:
        if df[col].isnull().any():
            if scaler is not None and hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
                col_index = np.where(scaler.feature_names_in_ == col)[0][0]
                fill_value = scaler.mean_[col_index]
            else:
                fill_value = df[col].mean() # Fallback for features not in scaler
            df[col] = df[col].fillna(fill_value)

    # Ensure that numerical_features are correctly ordered for transform
    df[numerical_features] = scaler.transform(df[numerical_features])

    print("Feature preprocessing completed for test data.")
    return df, encoded_categorical_data

# --- 6. PyTorch Dataset (用于测试数据) ---
class CustomUserDataset(TensorDataset): # Reused
    def __init__(self, numerical_data, categorical_data_dict, labels=None): # Labels might be None for test
        all_tensors = [numerical_data] + list(categorical_data_dict.values())
        if labels is not None:
            all_tensors.append(labels)
        super(CustomUserDataset, self).__init__(*all_tensors)
        self.categorical_keys = list(categorical_data_dict.keys())
        self._num_numerical_features = numerical_data.shape[1]
        self.has_labels = (labels is not None)

    def __getitem__(self, idx):
        item_tensors = super(CustomUserDataset, self).__getitem__(idx)
        numerical_feats = item_tensors[0]
        categorical_feats_item_dict = {}
        for i, key in enumerate(self.categorical_keys):
            categorical_feats_item_dict[key] = item_tensors[1 + i]
        
        if self.has_labels:
            label = item_tensors[-1]
            return numerical_feats, categorical_feats_item_dict, label
        else:
            return numerical_feats, categorical_feats_item_dict # No label for prediction

# --- 7. 深度学习模型定义 (与训练代码保持一致) ---
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

# --- 主预测流程 ---
if __name__ == '__main__':
    print("--- Starting Prediction Pipeline ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for prediction: {device}")
    
    BATCH_SIZE_PREDICT = 2048 # Use same or slightly larger batch size for prediction

    # 1. 加载训练时保存的预处理器和特征配置
    try:
        scaler_loaded = joblib.load(os.path.join(OUTPUT_DIR, 'scaler_final.joblib'))
        label_encoders_loaded = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoders_final.joblib'))
        feature_config = joblib.load(os.path.join(OUTPUT_DIR, 'feature_config.joblib'))
    except FileNotFoundError:
        print("Error: Pre-trained scaler, label encoders, or feature config not found.")
        print(f"Please ensure {OUTPUT_DIR} contains 'scaler_final.joblib', 'label_encoders_final.joblib', and 'feature_config.joblib' from training.")
        exit()

    final_numerical_features = feature_config['final_numerical_features']
    final_categorical_features = feature_config['final_categorical_features']
    embedding_dims_map_for_model = feature_config['embedding_dims_map']
    f1_threshold_prediction = feature_config['f1_threshold_prediction']

    print("Loaded pre-processors and feature configuration.")
    print(f"Features used by model: Numerical={final_numerical_features}, Categorical={final_categorical_features}")

    # 2. 加载测试数据，并保存原始DID和其在原始文件中的顺序
    df_test_original = pd.read_csv(TEST_DATA_PATH)
    original_dids_for_submission = df_test_original['did'].copy() # 保存原始DID，确保顺序
    
    # 对要处理的DataFrame进行复制，避免修改原始df_test_original
    df_test_processed = df_test_original.copy()

    # 3. 对测试数据进行特征工程 (与训练时保持一致)
    df_test_processed = feature_engineer_for_predict(df_test_processed)

    # 4. 对测试数据进行预处理（transform）
    # 确保df_test_processed只包含模型需要的特征，并且顺序与训练时一致
    X_test_for_predict_df = df_test_processed[final_numerical_features + final_categorical_features].copy()
    
    # 传递 is_train_mode=False
    X_test_processed_final_df, encoded_cat_test_data = \
        preprocess_features_for_predict(X_test_for_predict_df, final_numerical_features, final_categorical_features,
                                        scaler=scaler_loaded, label_encoders=label_encoders_loaded, is_train_mode=False)

    # 转换为 PyTorch 张量
    X_test_num_tensor = torch.tensor(X_test_processed_final_df[final_numerical_features].values, dtype=torch.float32)
    categorical_test_tensors = {
        col: torch.tensor(encoded_cat_test_data[col], dtype=torch.long)
        for col in final_categorical_features
    }

    # 创建 DataLoader (不带 labels)
    test_dataset = CustomUserDataset(X_test_num_tensor, categorical_test_tensors, labels=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))

    # 5. 加载模型
    model = NewUserPredictor(
        num_numerical_features=len(final_numerical_features),
        embedding_dims=embedding_dims_map_for_model
    ).to(device)

    # 寻找并加载最佳模型
    best_model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('best_model_f1_') and f.endswith('.pth')]
    if not best_model_files:
        print("Error: No best F1 model found to load. Please ensure training completed successfully.")
        exit()
    # 找出F1分数最高的模型文件
    best_model_filename = ''
    max_f1_in_filename = -1.0
    for f in best_model_files:
        try:
            f1_str = f.split('_f1_')[1].split('_epoch')[0]
            f1_val = float(f1_str)
            if f1_val > max_f1_in_filename:
                max_f1_in_filename = f1_val
                best_model_filename = f
        except Exception as e:
            print(f"Warning: Could not parse F1 from filename {f}: {e}")
            continue

    if not best_model_filename:
        print("Error: No valid best F1 model filename found for loading.")
        exit()
        
    best_model_path = os.path.join(OUTPUT_DIR, best_model_filename)
    
    print(f"Loading model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # 6. 进行预测
    print("\n--- Starting Prediction ---")
    all_predictions_prob = []
    with torch.no_grad():
        for numerical_data, categorical_data_dict_batch in test_loader:
            numerical_data = numerical_data.to(device)
            categorical_data_dict_batch_on_device = {
                k: v.to(device) for k, v in categorical_data_dict_batch.items()
            }
            outputs = model(numerical_data, categorical_data_dict_batch_on_device)
            probabilities = torch.sigmoid(outputs)
            all_predictions_prob.extend(probabilities.cpu().numpy().flatten())

    all_predictions_binary = (np.array(all_predictions_prob) > f1_threshold_prediction).astype(int)

    # 7. 生成提交文件，确保DID顺序与原始文件一致
    submission_df = pd.DataFrame({'did': original_dids_for_submission, 'is_new_did': all_predictions_binary})
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    
    print(f"\nPrediction completed. Submission file saved to: {SUBMISSION_FILE_PATH}")
    print("--- End of Prediction Script ---")