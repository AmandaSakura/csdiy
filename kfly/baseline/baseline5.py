# ==============================================================================
# BASELINE 5 (Corrected) - Sequential & Behavioral Features
# ==============================================================================
# - FIX: Preserves the original test set's 'did' order to ensure a valid submission.
# - This version introduces sequential and relative time features.
# - Aims to push F1 score beyond the 0.88 mark.
# ==============================================================================

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report
from datetime import datetime
import warnings
import optuna
import os
import pickle

# 确保 imbalanced-learn 已安装
try:
    from imblearn.under_sampling import RandomUnderSampler
    print("✅ imbalanced-learn 库已成功导入。")
except ImportError:
    print("❌ 错误: 未找到 imbalanced-learn 库。请先安装：pip install -U imbalanced-learn")
    exit()

warnings.filterwarnings('ignore')

# --- 0. 全局配置 (Global Configuration) ---
DATA_DIR = "/home/joker/new_csdiylearning2/kfly/data"
SAVE_DIR = "/home/joker/new_csdiylearning2/kfly/data/baseline5_output" # 为新baseline创建目录

UNDERSAMPLING_RATIO = 0.5
OPTUNA_TRIALS = 100
N_FOLDS = 5

os.makedirs(SAVE_DIR, exist_ok=True)
storage_name = f"sqlite:///{os.path.join(SAVE_DIR, 'optuna_study_b5.db')}"
study_name = "lgbm_f1_b5_sequential_feats"

# --- 1. 核心功能函数 ---
def translateCsvToDf(filepath, dtypes=None):
    return pd.read_csv(filepath, dtype=dtypes)

def feature_engineering(train_df, test_df):
    """特征工程主函数 - 加入顺序和相对时间特征"""
    print("\n=== [1/5] 开始特征工程 (Baseline 5) ===")
    
    # 这里的 full_df 是为了计算全局统计量，它的顺序会被改变
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
    
    # 将创建好的特征合并回原始的DataFrame，以保留它们的原始顺序
    # 注意：我们不再使用 full_df 来构建 X 和 X_test，而是用原始的 train_df 和 test_df 去合并特征
    
    # 首先处理 udmap
    for df in [train_df, test_df]:
        df['botId'] = np.nan
        df['pluginId'] = np.nan
        for idx, udmap_str in df['udmap'].items():
            if pd.isna(udmap_str): continue
            try:
                udmap_dict = json.loads(udmap_str)
                if 'botId' in udmap_dict: df.loc[idx, 'botId'] = udmap_dict['botId']
                if 'pluginId' in udmap_dict: df.loc[idx, 'pluginId'] = udmap_dict['pluginId']
            except: continue

    # 构建一个包含所有新特征的DataFrame，以 did 和 common_ts 作为键
    # 我们只从 full_df 中提取新创建的列
    new_features_df = full_df[['did', 'common_ts', 'event_rank', 'time_since_first']]
    
    # 将新特征和全局统计特征合并回原始的 train_df 和 test_df
    # 这样可以保持 train_df 和 test_df 的原始行顺序不变
    train_processed = train_df.merge(new_features_df, on=['did', 'common_ts'], how='left')
    train_processed = train_processed.merge(did_stats, on='did', how='left')
    
    test_processed = test_df.merge(new_features_df, on=['did', 'common_ts'], how='left')
    test_processed = test_processed.merge(did_stats, on='did', how='left')

    for df in [train_processed, test_processed]:
        for col in ['botId', 'pluginId', 'first_eid']:
            if col in df.columns:
                df[col] = df[col].fillna(-1).astype('int32')

    print("✅ 特征工程完成 (Baseline 5)。")
    return train_processed, test_processed

def find_best_f1_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# --- 2. 环境自检与数据加载 ---
print("代码开始执行...")
try:
    lgb.train({'device': 'cuda'}, lgb.Dataset(np.random.rand(10,2), label=np.random.randint(0,2,10)), num_boost_round=1)
    GPU_AVAILABLE = True
    print("✅ GPU加速可用，将使用 'cuda' 设备。")
except Exception:
    GPU_AVAILABLE = False
    print("⚠️ GPU不可用，将使用CPU。")

print("正在加载数据...")
OringinTrainDataUrl = os.path.join(DATA_DIR, "train_data/train.csv")
OringinTestDataUrl = os.path.join(DATA_DIR, "test_data/testA_data.csv")
train_df = translateCsvToDf(OringinTrainDataUrl)
test_df = translateCsvToDf(OringinTestDataUrl)

# <-- [修正点1] 在任何处理之前，备份原始测试集的did及其顺序
original_test_submission_df = test_df[['did']].copy()

# --- 3. 执行特征工程与数据准备 ---
train_df, test_df = feature_engineering(train_df, test_df)

exclude_features = ['did', 'udmap', 'common_ts', 'common_ts_dt', 'common_ts_min', 'common_ts_max', 'is_new_did']
feature_cols = [col for col in train_df.columns if col not in exclude_features]
X = train_df[feature_cols]
y = train_df['is_new_did']
X_test = test_df[feature_cols]

for col in X.select_dtypes(['int8', 'int16', 'int32']).columns:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print(f"模型将使用 {len(feature_cols)} 个特征: {feature_cols}")

# --- 4. Optuna 自动调参 (可中断并续跑) ---
# ... (此部分与之前完全相同，为保持完整性而保留) ...
print(f"\n=== [2/5] 开始或继续Optuna调参（存入 {storage_name}） ===")
def objective(trial):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'verbosity': -1, 'random_state': 42, 'device': 'cuda' if GPU_AVAILABLE else 'cpu',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_f1_scores = []
    for train_idx, val_idx in skf_opt.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLING_RATIO, random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)
        model = lgb.train(params, lgb.Dataset(X_train_resampled, label=y_train_resampled, categorical_feature='auto'),
                          valid_sets=[lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature='auto')],
                          num_boost_round=1500, callbacks=[lgb.early_stopping(100, verbose=False)])
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        _, best_f1 = find_best_f1_threshold(y_val_fold, val_pred)
        cv_f1_scores.append(best_f1)
    return np.mean(cv_f1_scores)

study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), load_if_exists=True)
if len(study.trials) < OPTUNA_TRIALS:
    study.optimize(objective, n_trials=(OPTUNA_TRIALS - len(study.trials)))

print("\n✅ 优化完成！")
best_params = study.best_params
print("最佳参数:", best_params)

# --- 5. 最终模型训练与评估 ---
print(f"\n=== [3/5] 使用最佳参数进行 {N_FOLDS}-折交叉验证训练 ===")
best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 42, 'device': 'cuda' if GPU_AVAILABLE else 'cpu'})
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
final_test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame(index=feature_cols)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLING_RATIO, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)
    model = lgb.train(best_params, lgb.Dataset(X_train_resampled, label=y_train_resampled, categorical_feature='auto'),
                      valid_sets=[lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature='auto')],
                      valid_names=['valid'], num_boost_round=3000, callbacks=[lgb.early_stopping(200, verbose=1000)])
    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    final_test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

# --- 6. 评估与结果保存 ---
print("\n=== [4/5] 开始模型评估 ===")
oof_best_threshold, oof_best_f1 = find_best_f1_threshold(y, oof_preds)
print(f"OOF 最佳F1分数: {oof_best_f1:.6f} (在阈值 {oof_best_threshold:.4f} 时取得)")
print("\nOOF 分类报告:")
print(classification_report(y, (oof_preds >= oof_best_threshold).astype(int), digits=4))

print("\n=== [5/5] 开始生成并保存结果 ===")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# <-- [修正点2] 使用原始顺序的 a DataFrame 来构建提交文件
submission = original_test_submission_df
submission['is_new_did'] = (final_test_preds >= oof_best_threshold).astype(int)
submission_filename = os.path.join(SAVE_DIR, f"submission_b5_corrected_{timestamp}_f1_{oof_best_f1:.4f}.csv")
submission.to_csv(submission_filename, index=False)
print(f"📄 预测结果已保存至: {submission_filename}")

# ... (保存特征重要性和研究对象等) ...
feature_importance['mean'] = feature_importance.mean(axis=1)
feature_importance.sort_values('mean', ascending=False, inplace=True)
feature_importance_filename = os.path.join(SAVE_DIR, f"feature_importance_b5_{timestamp}.csv")
feature_importance.to_csv(feature_importance_filename)
print(f"📄 特征重要性已保存至: {feature_importance_filename}")

study_filename = os.path.join(SAVE_DIR, f"optuna_study_b5_object_{timestamp}.pkl")
with open(study_filename, 'wb') as f:
    pickle.dump(study, f)
print(f"📄 Optuna研究对象已保存至: {study_filename}")

print("\n🎉 全部流程执行完毕！")