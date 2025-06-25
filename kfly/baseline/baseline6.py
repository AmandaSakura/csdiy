# ==============================================================================
# BASELINE 6 - Pseudo-Labeling
# ==============================================================================
# - Implements Pseudo-Labeling, a semi-supervised learning technique.
# - STAGE 1: Train a strong base model on the original data.
# - STAGE 2: Use the base model to predict on the test set and generate pseudo-labels
#            for high-confidence predictions.
# - STAGE 3: Combine original data with pseudo-labeled data.
# - STAGE 4: Retrain a final model on the enriched dataset.
# ==============================================================================

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report
from datetime import datetime
import warnings
import os

# 确保 imbalanced-learn 已安装
try:
    from imblearn.under_sampling import RandomUnderSampler
    print("✅ imbalanced-learn 库已成功导入。")
except ImportError:
    print("❌ 错误: 未找到 imbalanced-learn 库。请先安装：pip install -U imbalanced-learn")
    exit()

warnings.filterwarnings('ignore')

# --- 0. 全局配置 (Global Configuration) ---
# 路径配置
DATA_DIR = "/home/joker/new_csdiylearning2/kfly/data"
SAVE_DIR = "/home/joker/new_csdiylearning2/kfly/data/baseline6" # <--- 已更新为 baseline6 路径
os.makedirs(SAVE_DIR, exist_ok=True)

# 模型与伪标签配置
UNDERSAMPLING_RATIO = 0.5
N_FOLDS = 5
PSEUDO_LABEL_THRESH_HIGH = 0.99  # 高于此概率的测试集样本将被赋予 '1' 的伪标签
PSEUDO_LABEL_THRESH_LOW = 0.01   # 低于此概率的测试集样本将被赋予 '0' 的伪标签

# 使用您在 baseline5 中找到的最佳参数，如果需要重新调优，可以引入Optuna
# 这里我们直接使用已知最优参数来演示伪标签流程
BEST_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'device': 'cuda',
    # 以下参数应替换为您在baseline5中找到的最佳参数
    'num_leaves': 150, 'learning_rate': 0.1621265071250267, 
    'feature_fraction': 0.6986396182085532, 'bagging_fraction': 0.7310681715909233,
    'bagging_freq': 3, 'min_child_samples': 65, 
    'lambda_l1': 0.35743521920907556, 'lambda_l2': 0.03169483892100607
}

# --- 1. 核心功能函数 (与baseline5相同) ---
def translateCsvToDf(filepath, dtypes=None): return pd.read_csv(filepath, dtype=dtypes)
def feature_engineering(train_df, test_df):
    """特征工程主函数 - 使用 baseline5 的版本"""
    print("\n=== [STAGE 0/4] 开始特征工程 (沿用Baseline5) ===")
    full_df = pd.concat([train_df.drop('is_new_did', axis=1, errors='ignore'), test_df], ignore_index=True)
    full_df['common_ts_dt'] = pd.to_datetime(full_df['common_ts'], unit='ms')
    full_df = full_df.sort_values(by=['did', 'common_ts_dt']).reset_index(drop=True)
    did_stats = full_df.groupby('did').agg(
        mid_count=('mid', 'count'), eid_nunique=('eid', 'nunique'),
        common_ts_min=('common_ts_dt', 'first'), common_ts_max=('common_ts_dt', 'last'),
        first_eid=('eid', 'first')
    ).reset_index()
    did_stats['ts_span'] = (did_stats['common_ts_max'] - did_stats['common_ts_min']).dt.total_seconds()
    full_df['event_rank'] = full_df.groupby('did').cumcount() + 1
    full_df = full_df.merge(did_stats[['did', 'common_ts_min', 'first_eid']], on='did', how='left')
    full_df['time_since_first'] = (full_df['common_ts_dt'] - full_df['common_ts_min']).dt.total_seconds()
    def process_chunk(df, full_processed_df):
        df['botId'] = np.nan; df['pluginId'] = np.nan
        for idx, udmap_str in df['udmap'].items():
            if pd.isna(udmap_str): continue
            try:
                udmap_dict = json.loads(udmap_str)
                if 'botId' in udmap_dict: df.loc[idx, 'botId'] = udmap_dict['botId']
                if 'pluginId' in udmap_dict: df.loc[idx, 'pluginId'] = udmap_dict['pluginId']
            except: continue
        df = df.reset_index().merge(full_processed_df, on=['did', 'common_ts'], how='left', suffixes=('', '_y')).set_index('index')
        df.drop([col for col in df.columns if '_y' in col], axis=1, inplace=True)
        for col in ['botId', 'pluginId', 'first_eid']:
            if col in df.columns: df[col] = df[col].fillna(-1).astype('int32')
        return df
    features_to_merge = full_df[['did', 'common_ts', 'event_rank', 'first_eid', 'time_since_first']]
    train_processed = process_chunk(train_df, features_to_merge).merge(did_stats.drop(['common_ts_min', 'common_ts_max', 'first_eid'], axis=1), on='did', how='left')
    test_processed = process_chunk(test_df, features_to_merge).merge(did_stats.drop(['common_ts_min', 'common_ts_max', 'first_eid'], axis=1), on='did', how='left')
    print("✅ 特征工程完成。")
    return train_processed, test_processed
def find_best_f1_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# --- 2. 数据加载与准备 ---
print("代码开始执行...")
OringinTrainDataUrl = os.path.join(DATA_DIR, "train_data/train.csv")
OringinTestDataUrl = os.path.join(DATA_DIR, "test_data/testA_data.csv")
train_df = translateCsvToDf(OringinTrainDataUrl)
test_df = translateCsvToDf(OringinTestDataUrl)

train_df, test_df = feature_engineering(train_df, test_df)

exclude_features = ['did', 'udmap', 'common_ts', 'common_ts_dt', 'is_new_did']
feature_cols = [col for col in train_df.columns if col not in exclude_features]
X = train_df[feature_cols]
y = train_df['is_new_did']
X_test = test_df[feature_cols]

for col in X.select_dtypes(['int8', 'int16', 'int32']).columns:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# --- 3. 伪标签流程 ---
# STAGE 1: 训练初始模型并生成OOF预测（用于后续评估）和测试集预测（用于生成伪标签）
print("\n=== [STAGE 1/4] 训练初始模型 ===")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
initial_oof_preds = np.zeros(len(X))
initial_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- 初始模型 Fold {fold + 1}/{N_FOLDS} ---")
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLING_RATIO, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)
    
    model = lgb.train(BEST_PARAMS, lgb.Dataset(X_train_resampled, label=y_train_resampled, categorical_feature='auto'),
                      num_boost_round=3000, callbacks=[lgb.early_stopping(200, verbose=False)])
    
    initial_oof_preds[val_idx] = model.predict(X_val_fold)
    initial_test_preds += model.predict(X_test) / N_FOLDS

# STAGE 2: 生成伪标签
print("\n=== [STAGE 2/4] 生成伪标签 ===")
pseudo_labels_high = test_df[initial_test_preds > PSEUDO_LABEL_THRESH_HIGH].copy()
pseudo_labels_high['is_new_did'] = 1
pseudo_labels_low = test_df[initial_test_preds < PSEUDO_LABEL_THRESH_LOW].copy()
pseudo_labels_low['is_new_did'] = 0

pseudo_df = pd.concat([pseudo_labels_high, pseudo_labels_low], ignore_index=True)
print(f"✅ 生成了 {len(pseudo_df)} 条高置信度伪标签 (正例: {len(pseudo_labels_high)}, 负例: {len(pseudo_labels_low)})")

# STAGE 3: 合并数据
print("\n=== [STAGE 3/4] 合并原始数据与伪标签数据 ===")
X_pseudo = pseudo_df[feature_cols]
y_pseudo = pseudo_df['is_new_did']

# 转换伪标签数据的类别特征
for col in X_pseudo.select_dtypes(['int8', 'int16', 'int32']).columns:
    X_pseudo[col] = X_pseudo[col].astype('category')

X_combined = pd.concat([X, X_pseudo], ignore_index=True)
y_combined = pd.concat([y, y_pseudo], ignore_index=True)
print(f"新训练集大小: {X_combined.shape}, 类别分布:\n{y_combined.value_counts()}")

# STAGE 4: 在增强数据集上重新训练最终模型
print("\n=== [STAGE 4/4] 在增强数据集上重新训练最终模型 ===")
final_test_preds = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
    print(f"--- 最终模型 Fold {fold + 1}/{N_FOLDS} ---")
    X_train_fold, y_train_fold = X_combined.iloc[train_idx], y_combined.iloc[train_idx]
    
    # 同样进行欠采样
    rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLING_RATIO, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)
    
    model = lgb.train(BEST_PARAMS, lgb.Dataset(X_train_resampled, label=y_train_resampled, categorical_feature='auto'),
                      num_boost_round=3000, callbacks=[lgb.early_stopping(200, verbose=False)])
    
    final_test_preds += model.predict(X_test) / N_FOLDS

# --- 4. 评估与保存 ---
print("\n=== 模型评估 (基于原始数据的OOF) ===")
oof_best_threshold, oof_best_f1 = find_best_f1_threshold(y, initial_oof_preds)
print(f"初始模型OOF F1分数: {oof_best_f1:.6f} (在阈值 {oof_best_threshold:.4f} 时取得)")
print("\n初始模型OOF 分类报告:")
print(classification_report(y, (initial_oof_preds >= oof_best_threshold).astype(int), digits=4))

print("\n=== 生成并保存最终结果 ===")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({'did': test_df['did']})
# 使用初始模型OOF计算出的最佳阈值
submission['is_new_did'] = (final_test_preds >= oof_best_threshold).astype(int)
submission_filename = os.path.join(SAVE_DIR, f"submission_b6_pseudolabel_{timestamp}.csv")
submission.to_csv(submission_filename, index=False)
print(f"📄 预测结果已保存至: {submission_filename}")
print("预测结果类别分布:\n", submission['is_new_did'].value_counts(normalize=True))

print("\n🎉 全部流程执行完毕！")