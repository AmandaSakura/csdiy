import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os
import joblib
import gc
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. 配置和路径定义 ---
script_dir = os.path.dirname(__file__)

TEST_DATA_PATH = os.path.join(script_dir, '..', 'data', 'test_data', 'test.csv')
OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'baseline10_output') # 与训练脚本保持一致
SUBMISSION_DIR = os.path.join(script_dir, '..', 'submission')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model_f1_*.pth') # 载入最佳模型，需要匹配训练保存的格式
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler_final.joblib')
LABEL_ENCODERS_PATH = os.path.join(OUTPUT_DIR, 'label_encoders_final.joblib')
VECTORIZERS_PATH = os.path.join(OUTPUT_DIR, 'vectorizers_final.joblib') # 载入文本向量化器
FEATURE_CONFIG_PATH = os.path.join(OUTPUT_DIR, 'feature_config.joblib')

BATCH_SIZE = 2048 # 测试集也可以用大批量
UNSEEN_CATEGORY_PLACEHOLDER = '___UNSEEN___'
UNSEEN_UDMAP_KEY_PLACEHOLDER = 'UNSEEN_KEY'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# --- 2. 辅助函数：处理 udmap 字段 (与训练代码一致) ---
def parse_udmap_extended_helper(udmap_str):
    parsed_data = {'botId': 'unknown_bot', 'pluginId': 'unknown_plugin', 'udmap_keys_str': ''}
    if isinstance(udmap_str, str) and udmap_str != '{}':
        try:
            data = json.loads(udmap_str)
            parsed_data['botId'] = str(data.get('botId', 'unknown_bot'))
            parsed_data['pluginId'] = str(data.get('pluginId', 'unknown_plugin'))
            parsed_data['udmap_keys_str'] = ' '.join(sorted(data.keys()))
        except json.JSONDecodeError:
            pass
    return parsed_data

# --- 3. 数据加载与基础清洗 (与训练代码一致) ---
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

    udmap_parsed_series = df['udmap'].apply(parse_udmap_extended_helper)
    df_udmap_features = pd.json_normalize(udmap_parsed_series)
    df = pd.concat([df.drop(columns=['udmap']), df_udmap_features], axis=1)
    
    categorical_id_cols.extend(['botId', 'pluginId'])
    
    for col in categorical_id_cols:
        if col in df.columns:
            df[col] = df[col].fillna(f'unknown_{col}').astype(str)

    df['did'] = df['did'].astype(str)
    return df, categorical_id_cols

# --- 4. 特征工程 (与训练代码一致) ---
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

    # 移除训练中删除的列，只保留用于建模的列
    df_to_drop_cols = ['common_ts', 'common_ts_dt', 'common_ts_min'] # 'did' 在加载时已经保留，但在特征工程结束时才删除
    df = df.drop(columns=[col for col in df_to_drop_cols if col in df.columns])
    
    print("Feature engineering completed.")
    return df

# --- 5. 特征编码与标准化 (与训练代码一致，但is_train_mode=False) ---
def preprocess_features(df, numerical_features, categorical_features, text_features,
                        scaler, label_encoders, vectorizers, is_train_mode=False):
    print("Starting feature preprocessing (encoding and scaling)...")

    encoded_categorical_data = {}
    embedding_dims_map = {} # 测试模式下，这个map可以从训练时的feature_config中加载

    for col in categorical_features:
        if col not in label_encoders:
            print(f"Warning: LabelEncoder for {col} not found. Skipping encoding.")
            continue # 如果没有找到对应的编码器，跳过
            
        unseen_mask = ~df[col].astype(str).isin(label_encoders[col].classes_)
        df.loc[unseen_mask, col] = UNSEEN_CATEGORY_PLACEHOLDER
        df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
        df[f'{col}_encoded'] = df[f'{col}_encoded'].astype(int)
        encoded_categorical_data[col] = df[f'{col}_encoded'].values

    # --- 处理文本特征 ---
    text_numerical_features_names = []
    for col in text_features:
        if col not in vectorizers:
            print(f"Warning: CountVectorizer for {col} not found. Skipping text feature processing.")
            continue
            
        # 填充测试集中的未知值，确保它们在词汇表中存在或映射到未知
        # 先处理NaN，再处理未知词汇
        df[col] = df[col].fillna(UNSEEN_UDMAP_KEY_PLACEHOLDER)
        # 将测试集中不在训练集词汇表中的词替换为placeholder
        # (CountVectorizer 会自动处理未见词，将其忽略。这里主要是为了统一处理NaN)
        
        text_matrix = vectorizers[col].transform(df[col])
        
        text_df = pd.DataFrame(text_matrix.toarray(), columns=[f'{col}_text_{i}' for i in range(text_matrix.shape[1])], index=df.index)
        df = pd.concat([df, text_df], axis=1)
        text_numerical_features_names.extend(text_df.columns.tolist())

    # --- 数值特征标准化 ---
    all_numerical_features_to_scale = numerical_features + text_numerical_features_names
    # 确保只对训练时scaler中包含的特征进行标准化
    # 如果 scaler.feature_names_in_ 存在，则只用这些特征
    if hasattr(scaler, 'feature_names_in_'):
        cols_to_transform = [col for col in all_numerical_features_to_scale if col in scaler.feature_names_in_]
        # 对测试集数据中，如果存在训练集中没有的列，需要处理
        missing_in_scaler = set(all_numerical_features_to_scale) - set(scaler.feature_names_in_)
        if missing_in_scaler:
            print(f"Warning: Some numerical features in test set not found in scaler: {missing_in_scaler}. These will not be scaled.")
            # 对于这些缺失的列，保持原样或填充0
            for m_col in missing_in_scaler:
                if m_col not in df.columns: # 确保列存在
                     df[m_col] = 0.0 # 填充0

        # 对所有数值特征列进行缺失值填充，使用训练集学到的均值
        for col in cols_to_transform:
            if df[col].isnull().any():
                if col in scaler.feature_names_in_:
                    col_index = np.where(scaler.feature_names_in_ == col)[0][0]
                    fill_value = scaler.mean_[col_index]
                    df[col] = df[col].fillna(fill_value)
                else: # Fallback if for some reason feature not in scaler but in test_numerical_features
                    df[col] = df[col].fillna(df[col].mean()) # Fallback to local mean

        df[cols_to_transform] = scaler.transform(df[cols_to_transform])
    else: # Fallback if scaler doesn't have feature_names_in_ (should not happen for fitted scaler)
        print("Warning: Scaler does not have 'feature_names_in_'. Proceeding with full numerical features.")
        for col in all_numerical_features_to_scale:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean()) # Fallback to local mean

        df[all_numerical_features_to_scale] = scaler.transform(df[all_numerical_features_to_scale])


    print("Feature preprocessing completed.")
    return df, encoded_categorical_data, text_numerical_features_names

# --- 6. PyTorch Dataset 和 DataLoader (与训练代码一致) ---
class CustomUserDataset(TensorDataset):
    def __init__(self, numerical_data, categorical_data_dict, labels=None):
        if numerical_data.dim() == 1:
            numerical_data = numerical_data.unsqueeze(1)
        
        self.categorical_keys_ordered = sorted(categorical_data_dict.keys())
        all_categorical_tensors_list = [categorical_data_dict[key] for key in self.categorical_keys_ordered]

        if labels is not None:
            all_tensors = [numerical_data] + all_categorical_tensors_list + [labels]
        else: # For test set, no labels
            all_tensors = [numerical_data] + all_categorical_tensors_list

        super(CustomUserDataset, self).__init__(*all_tensors)
        
        self._num_numerical_features = numerical_data.shape[1]
        self.has_labels = (labels is not None)

    def __getitem__(self, idx):
        item_tensors = super(CustomUserDataset, self).__getitem__(idx)
        numerical_feats = item_tensors[0]
        categorical_feats_item_dict = {}
        
        for i, key in enumerate(self.categorical_keys_ordered):
            categorical_feats_item_dict[key] = item_tensors[1 + i]
        
        if self.has_labels:
            label = item_tensors[-1]
            return numerical_feats, categorical_feats_item_dict, label
        else:
            return numerical_feats, categorical_feats_item_dict

# --- 7. 深度学习模型定义 (与训练代码一致) ---
class NewUserPredictor(nn.Module):
    def __init__(self, num_numerical_features, embedding_dims):
        super(NewUserPredictor, self).__init__()

        self.embeddings = nn.ModuleDict()
        embedding_output_dim = 0
        for col, (num_categories, embed_dim) in embedding_dims.items():
            self.embeddings[col] = nn.Embedding(num_categories, embed_dim)
            embedding_output_dim += embed_dim

        self.total_input_dim = num_numerical_features + embedding_output_dim

        self.fc1 = nn.Linear(self.total_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(128, 1)

    def forward(self, numerical_feats, categorical_feats_dict):
        embedded_feats = []
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

# --- 主执行流程 ---
if __name__ == '__main__':
    print("--- Starting Inference Pipeline ---")

    # 1. 加载模型、Scaler、LabelEncoders 和 Feature Config
    try:
        # 查找最新的模型文件
        list_of_files = glob.glob(MODEL_PATH)
        if not list_of_files:
            raise FileNotFoundError(f"No model file found matching {MODEL_PATH}")
        latest_model_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading model from: {latest_model_file}")
        
        feature_config = joblib.load(FEATURE_CONFIG_PATH)
        scaler_final = joblib.load(SCALER_PATH)
        label_encoders_final = joblib.load(LABEL_ENCODERS_PATH)
        vectorizers_final = joblib.load(VECTORIZERS_PATH) # 载入文本向量化器

        final_numerical_features = feature_config['final_numerical_features']
        final_categorical_features = feature_config['final_categorical_features']
        embedding_dims_map = feature_config['embedding_dims_map']
        f1_threshold_prediction = feature_config['f1_threshold_prediction']
        final_text_numerical_features_names = feature_config.get('final_text_numerical_features_names', []) # 获取文本数值特征名

        model = NewUserPredictor(
            num_numerical_features=len(final_numerical_features),
            embedding_dims=embedding_dims_map
        ).to(device)
        model.load_state_dict(torch.load(latest_model_file, map_location=device))
        model.eval() # 设置为评估模式
        print("Model, scaler, encoders, vectorizers and feature config loaded successfully.")
        print(f"Using F1 prediction threshold: {f1_threshold_prediction:.4f}")

    except Exception as e:
        print(f"Error loading assets: {e}")
        exit() # 退出程序

    # 2. 加载和清洗测试数据
    df_test_raw, all_categorical_features_after_udmap = load_and_clean_data(TEST_DATA_PATH)
    test_dids = df_test_raw['did'].unique() # 保存 did，用于最终提交

    # 3. 特征工程
    df_test_engineered = feature_engineer(df_test_raw.copy()) # 注意：这里要确保不删除 did 列，直到最后

    # 为预处理准备特征列表，确保顺序一致
    # raw_numerical_features from training script (cols without target and udmap related)
    all_numerical_features_raw_test = [col for col in df_test_engineered.columns if df_test_engineered[col].dtype in ['int64', 'float64'] and col != 'did']
    # category features
    additional_cat_features = ['hour', 'dayofweek', 'month', 'year']
    all_categorical_features_test = list(set(all_categorical_features_after_udmap + additional_cat_features))
    # text features
    text_features = ['udmap_keys_str'] # 确保名称一致

    # 4. 预处理测试数据
    df_test_processed, encoded_cat_test_data, _ = \
        preprocess_features(df_test_engineered.copy(), all_numerical_features_raw_test, all_categorical_features_test, text_features,
                            scaler=scaler_final, label_encoders=label_encoders_final, vectorizers=vectorizers_final, is_train_mode=False)
    
    del df_test_engineered, df_test_raw
    gc.collect()

    # 确保用于 Tensor 的数值特征列与训练时一致
    final_numerical_cols_for_test_tensor = [col for col in final_numerical_features if col in df_test_processed.columns]
    X_test_num_tensor = torch.tensor(df_test_processed[final_numerical_cols_for_test_tensor].values, dtype=torch.float32)

    categorical_test_tensors = {
        col: torch.tensor(encoded_cat_test_data[col], dtype=torch.long)
        for col in final_categorical_features
    }
    
    # 5. 创建测试 DataLoader
    # 测试集没有标签，所以CustomUserDataset只需两个参数
    test_dataset = CustomUserDataset(X_test_num_tensor, categorical_test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=max(0, os.cpu_count() // 2 - 1))
    
    print(f"Test batches: {len(test_loader)}")

    # 6. 进行预测
    all_test_preds_prob = []
    with torch.no_grad():
        for batch_idx, (numerical_data, categorical_data_dict_batch) in enumerate(test_loader):
            numerical_data = numerical_data.to(device)
            categorical_data_dict_batch_on_device = {
                k: v.to(device) for k, v in categorical_data_dict_batch.items()
            }
            outputs = model(numerical_data, categorical_data_dict_batch_on_device)
            all_test_preds_prob.extend(torch.sigmoid(outputs).cpu().numpy())

    all_test_preds_prob = np.array(all_test_preds_prob).flatten()

    # 7. 生成提交文件
    submission_df = pd.DataFrame({'did': df_test_processed['did'].values, 'prob': all_test_preds_prob})
    
    # 因为训练集和测试集可能包含重复的did，但提交只需要每个did一个预测，取最后一个事件的did作为代表
    # 这里假设 competition scoring uses the latest prediction for a DID, or you need to aggregate
    # For now, let's just make sure each did has one prediction.
    # If the competition specifically requires the target to be for the *first* event of a new DID,
    # then the submission logic needs to be tied to the initial unique DIDs from the raw test file.
    # Assuming competition asks for `did` in `test.csv` (which are unique by row ID, not by actual `did` value)
    
    # Your current test.csv format might imply each row in test.csv is an event,
    # and you need to predict for EACH event whether it's a new did?
    # Or for each unique did in the test set?
    # If it's for EACH event, then submission_df is already correct.
    # If it's for unique dids, then group by did and take a single prediction per did (e.g. max prob or latest event prob).
    
    # Assuming `did` in submission should correspond to the `did` column in the processed test_df,
    # which has one row per event. The task "is_new_did" is typically associated with the DID itself,
    # so we might need to aggregate predictions per DID.
    
    # If the `test.csv` actually contains unique `did` for which you need to predict `is_new_did`
    # and your model predicts per event, you need to aggregate per DID for submission.
    # If test.csv provides `id` and `did` for each event, and expects `id` and `is_new_did` prediction.
    
    # Let's assume for now, the submission needs a prediction per 'id' in test.csv.
    # You need to merge with the original test_data to get the 'id' column.
    
    # Re-loading test.csv just to get 'id' column to map back to predictions
    df_original_test = pd.read_csv(TEST_DATA_PATH)
    
    # Ensure df_test_processed has 'id' and 'did' columns (which it will if it came from df_test_raw)
    submission_df = df_test_processed[['id', 'did']].copy()
    submission_df['prob'] = all_test_preds_prob

    # Final prediction is binary based on the learned threshold
    submission_df['is_new_did'] = (submission_df['prob'] > f1_threshold_prediction).astype(int)

    # Reorder columns as required for submission (assuming 'id' then 'is_new_did')
    submission_df = submission_df[['id', 'is_new_did']]

    submission_file_name = os.path.join(SUBMISSION_DIR, 'submission_baseline10.csv')
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file saved to {submission_file_name}")

    print("\n--- Inference Finished ---")