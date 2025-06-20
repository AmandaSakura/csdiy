import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed

# 忽略一些pandas的警告
warnings.filterwarnings('ignore')

# ---------------------------------------------
# --- 1. 内存优化与数据加载 (终极优化版) ---
# ---------------------------------------------

def reduce_mem_usage(df, verbose=True):
    """
    通过向下转换数据类型来减少DataFrame的内存使用量。
    """
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

print("Step 1: Inferring optimal dtypes from a data sample...")
# 定义文件路径
train_path = '/home/joker/new_csdiylearning2/kfly/data/train_data/train.csv'
test_path = '/home/joker/new_csdiylearning2/kfly/data/test_data/testA_data.csv'

# 通过加载一个小样本来推断最佳数据类型
# 增加样本量以提高推断的准确性
sample_df = pd.read_csv(train_path, nrows=100000)
sample_df = reduce_mem_usage(sample_df)

# 获取优化后的数据类型，并转换为字典
# 对于object类型，我们将在加载时使用 'category'，这是一种高效的内存优化方式
optimal_dtypes = sample_df.dtypes.to_dict()
for col, dtype in optimal_dtypes.items():
    if dtype == 'object':
        optimal_dtypes[col] = 'category'

# is_new_did 在测试集中不存在，加载测试集时要移除此键
if 'is_new_did' in optimal_dtypes:
    optimal_dtypes_test = optimal_dtypes.copy()
    optimal_dtypes_test.pop('is_new_did')
else:
    optimal_dtypes_test = optimal_dtypes

print("\nOptimal dtypes inferred. Now loading full datasets with optimized types...")

# 使用推断出的最佳数据类型加载完整数据集，从源头节省内存
train_df = pd.read_csv(train_path, dtype=optimal_dtypes)
test_df = pd.read_csv(test_path, dtype=optimal_dtypes_test)
sub_df = test_df[['did']].copy()

# 再次运行reduce_mem_usage确保万无一失
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# 合并进行统一的特征工程
data_df = pd.concat([train_df, test_df], ignore_index=True)
del train_df, test_df
print("Data loaded successfully using optimal dtypes.")

# -----------------------
# --- 2. 特征工程 ---
# -----------------------
print("\nStep 2: Feature Engineering...")

def parse_udmap(d):
    try:
        # 如果是category类型，需要先转成str
        if not isinstance(d, str): d = str(d)
        if d == '{}' or d == 'nan': return {}
        return json.loads(d)
    except: return {}

# udmap列现在是category类型，需要特殊处理
tqdm.pandas(desc="Parsing udmap")
udmap_df = data_df['udmap'].astype(str).progress_apply(parse_udmap).apply(pd.Series)
udmap_df = udmap_df.add_prefix('udmap_')
data_df = pd.concat([data_df, udmap_df], axis=1)

# udmap解析后可能产生object列，将其转换为category
for col in udmap_df.columns:
    if data_df[col].dtype == 'object':
        data_df[col] = data_df[col].astype('category')

data_df['common_ts'] = pd.to_datetime(data_df['common_ts'], unit='ms')
data_df['hour'] = data_df['common_ts'].dt.hour.astype(np.int8)
data_df['day'] = data_df['common_ts'].dt.day.astype(np.int8)
data_df['dayofweek'] = data_df['common_ts'].dt.dayofweek.astype(np.int8)
data_df['month'] = data_df['common_ts'].dt.month.astype(np.int8)

# 由于加载时已指定类型，大部分列已是数值或category，LabelEncoder主要用于统一编码
all_categorical_features = [col for col in data_df.columns if col not in ['is_new_did', 'did', 'common_ts']]

for col in tqdm(all_categorical_features, desc="Encoding Features"):
    if data_df[col].dtype.name not in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
        # 使用 .cat.codes 将 category 类型转换为整数编码
        data_df[col] = data_df[col].astype('category').cat.codes

data_df = data_df.drop(['udmap', 'common_ts'], axis=1)
# 对整个DataFrame再次进行内存优化
data_df = reduce_mem_usage(data_df)
print("Feature Engineering complete.")


# -----------------------
# --- 3. 准备训练数据 ---
# -----------------------
print("\nStep 3: Preparing data for training...")
train_processed_df = data_df[data_df['is_new_did'].notna()]
test_processed_df = data_df[data_df['is_new_did'].isna()]
del data_df

features = [col for col in train_processed_df.columns if col not in ['is_new_did', 'did']]
X_train = train_processed_df[features]
y_train = train_processed_df['is_new_did'].astype(np.int8)
X_test = test_processed_df[features]
del train_processed_df, test_processed_df

# ---------------------------------------------
# --- 4. 定义单折训练函数与并行交叉验证 ---
# ---------------------------------------------

def train_fold(fold, train_idx, val_idx, X_train, y_train, X_test, params, cat_features):
    """训练单一折模型的函数，将被并行调用。"""
    print(f"--- Starting Fold {fold+1} ---")
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='f1',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=[col for col in cat_features if col in X_train.columns])
    val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
    test_preds_fold = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_val_fold, val_preds_proba > t) for t in thresholds]
    best_f1 = np.max(f1_scores)
    print(f"--- Finished Fold {fold+1}, Best F1 Score: {best_f1:.5f} ---")
    return val_preds_proba, test_preds_fold, best_f1, val_idx

print("\nStep 4: Starting Parallel Model Training...")

params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'device': 'gpu', 'n_estimators': 2000, 'learning_rate': 0.02,
    'num_leaves': 31, 'seed': 42, 'n_jobs': 2, 'verbose': -1,
    'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum()
}
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

results = Parallel(n_jobs=-1)(
    delayed(train_fold)(
        fold, train_idx, val_idx, X_train, y_train, X_test, params, all_categorical_features
    ) for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))
)

# ------------------------------------
# --- 5. 汇总结果并生成提交文件 ---
# ------------------------------------
print("\nStep 5: Aggregating results and generating submission file...")

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_f1_scores = []

for val_preds_proba, test_preds_fold, best_f1, val_idx in results:
    oof_preds[val_idx] = val_preds_proba
    test_preds += test_preds_fold / N_SPLITS
    fold_f1_scores.append(best_f1)

print(f"\nAverage F1 Score across all folds: {np.mean(fold_f1_scores):.5f}")

thresholds = np.linspace(0.01, 0.99, 200)
global_f1_scores = [f1_score(y_train, oof_preds > t) for t in thresholds]
global_best_f1 = np.max(global_f1_scores)
global_best_threshold = thresholds[np.argmax(global_f1_scores)]

print(f"Overall OOF F1 Score: {global_best_f1:.5f} at best threshold: {global_best_threshold:.3f}")

final_predictions = (test_preds > global_best_threshold).astype(np.int8)
sub_df['is_new_did'] = final_predictions
sub_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print(f"Prediction distribution in submission file:\n{sub_df['is_new_did'].value_counts(normalize=True)}")