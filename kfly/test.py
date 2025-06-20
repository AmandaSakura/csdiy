import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import warnings

# 忽略一些pandas的警告
warnings.filterwarnings('ignore')

print("Step 1: Loading data...")
# --- 1. 数据加载 ---
# 定义你的文件路径
train_path = '/home/joker/new_csdiylearning2/kfly/data/train_data/train.csv'
test_path = '/home/joker/new_csdiylearning2/kfly/data/test_data/testA_data.csv'
# 提交文件的did需要从test文件中获取
sub_df = pd.read_csv(test_path)[['did']]


# 为了方便统一处理，我们先加载所有数据
# 注意：测试集没有 is_new_did 列
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 合并训练集和测试集，方便进行统一的特征工程
# 'is_new_did' 在测试集中是 NaN
data_df = pd.concat([train_df, test_df], ignore_index=True)

print("Data loaded successfully.")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Combined data shape: {data_df.shape}")


print("\nStep 2: Feature Engineering...")
# --- 2. 特征工程 ---

# 2.1 udmap JSON 解析
# tqdm用于显示处理进度条
def parse_udmap(d):
    try:
        # 如果是空的json "{}", loads之后是字典，否则可能是nan等
        if not isinstance(d, str) or d == '{}':
            return {}
        return json.loads(d)
    except:
        return {}

tqdm.pandas(desc="Parsing udmap")
udmap_df = data_df['udmap'].progress_apply(parse_udmap).apply(pd.Series)
# 填充NaN值，并将列名加上前缀以区分
udmap_df = udmap_df.add_prefix('udmap_')

# 创建新特征：udmap中key的数量
data_df['udmap_key_count'] = udmap_df['udmap'].apply(lambda x: len(parse_udmap(x)))

# 将解析出的udmap特征合并回主DataFrame
data_df = pd.concat([data_df, udmap_df], axis=1)

# 2.2 common_ts 时间戳处理
data_df['common_ts'] = pd.to_datetime(data_df['common_ts'], unit='ms')
data_df['hour'] = data_df['common_ts'].dt.hour
data_df['day'] = data_df['common_ts'].dt.day
data_df['dayofweek'] = data_df['common_ts'].dt.dayofweek # Monday=0, Sunday=6
data_df['month'] = data_df['common_ts'].dt.month

# 2.3 类别特征编码
# 识别所有非数值类型的列以及我们认为是分类的数值ID列
# 'is_new_did'是目标，'did'是ID，'common_ts'和'udmap'已经处理过了
categorical_features = [col for col in data_df.columns if data_df[col].dtype == 'object' and col not in ['is_new_did', 'did', 'common_ts', 'udmap']]
# 手动添加我们认为是分类的数值ID列
numerical_categorical_features = ['mid', 'eid', 'device_brand', 'ntt', 'operator', 'common_country',
                                  'common_province', 'common_city', 'appver', 'channel', 'os_type']
all_categorical_features = categorical_features + numerical_categorical_features

print(f"Categorical features to be encoded: {all_categorical_features}")

for col in tqdm(all_categorical_features, desc="Encoding Features"):
    # 使用 LabelEncoder，它能处理NaN值
    le = LabelEncoder()
    # 使用 .astype(str) 确保所有值（包括数字和NaN）都能被编码
    data_df[col] = le.fit_transform(data_df[col].astype(str))

# 删除不再需要的原始列
data_df = data_df.drop(['udmap', 'common_ts'], axis=1)

print("Feature Engineering complete.")
print(f"Data shape after FE: {data_df.shape}")


# --- 3. 准备训练数据 ---
# 将合并后的数据分离回训练集和测试集
train_processed_df = data_df[data_df['is_new_did'].notna()]
test_processed_df = data_df[data_df['is_new_did'].isna()]

# 定义特征列（X）和目标列（y）
features = [col for col in train_processed_df.columns if col not in ['is_new_did', 'did']]
X_train = train_processed_df[features]
y_train = train_processed_df['is_new_did']
X_test = test_processed_df[features]

print(f"\nTraining features: {features}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}")

# --- 4. 模型训练与交叉验证 ---
print("\nStep 3: Model Training with 5-Fold Stratified Cross-Validation...")

# LightGBM 模型参数
# 这是一个稳健的baseline参数集
params = {
    'objective': 'binary', # 二分类任务
    'metric': 'binary_logloss', # 评价指标，我们会在代码中单独计算F1
    'boosting_type': 'gbdt',
    'n_estimators': 2000, # 较大的树数量，配合early stopping
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8, # 特征采样
    'subsample': 0.8, # 数据采样
    'reg_alpha': 0.1, # L1 正则化
    'reg_lambda': 0.1, # L2 正则化
    # F1 Score对类别不平衡敏感，使用scale_pos_weight可以缓解
    'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum()
}

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train)) # 存储Out-of-Fold的预测概率
test_preds = np.zeros(len(X_test)) # 存储测试集的平均预测概率
fold_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='f1', # 使用f1作为early stopping的监控指标
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=[col for col in all_categorical_features if col in features]) # 告知LGBM哪些是类别特征

    # 预测验证集（用于OOF和评估）
    val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_preds_proba
    
    # 找到当前折的最佳阈值并计算F1
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_val_fold, val_preds_proba > t) for t in thresholds]
    best_f1 = np.max(f1_scores)
    best_threshold_fold = thresholds[np.argmax(f1_scores)]
    fold_f1_scores.append(best_f1)
    print(f"Fold {fold+1} Best F1 Score: {best_f1:.5f} at threshold {best_threshold_fold:.3f}")

    # 预测测试集，并将结果累加
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

print(f"\nAverage F1 Score across all folds: {np.mean(fold_f1_scores):.5f}")

# --- 5. 寻找最优阈值并生成提交文件 ---
print("\nStep 4: Finding optimal threshold and generating submission file...")

# 基于所有OOF预测结果找到全局最佳阈值
thresholds = np.linspace(0.01, 0.99, 200)
global_f1_scores = [f1_score(y_train, oof_preds > t) for t in thresholds]
global_best_f1 = np.max(global_f1_scores)
global_best_threshold = thresholds[np.argmax(global_f1_scores)]

print(f"Overall OOF F1 Score: {global_best_f1:.5f} at best threshold: {global_best_threshold:.3f}")

# 使用找到的最佳阈值对测试集预测结果进行二值化
final_predictions = (test_preds > global_best_threshold).astype(int)

# 创建提交文件
sub_df['is_new_did'] = final_predictions
sub_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print(f"Prediction distribution in submission file:\n{sub_df['is_new_did'].value_counts(normalize=True)}")