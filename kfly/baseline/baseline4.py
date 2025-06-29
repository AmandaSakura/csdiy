# ==============================================================================
# FINAL CORRECTED VERSION - Resumable Training Pipeline
# ==============================================================================
# - Corrected the GPU check logic by removing the deprecated 'verbose_eval' parameter.
# - This version should now correctly detect and utilize the GPU.
# - All other optimizations (Undersampling, Resumable Optuna, F1-Score objective) are retained.
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
# 路径配置
DATA_DIR = "/home/joker/new_csdiylearning2/kfly/data"
SAVE_DIR = "/home/joker/new_csdiylearning2/kfly/data/final_output" # 为最终版创建一个新目录

# 模型与调优配置
UNDERSAMPLING_RATIO = 0.5  # 多数类(老用户)数量是少数类(新用户)的2倍 (多数类:少数类 = 1:0.5)
OPTUNA_TRIALS = 100        # Optuna 优化的总试验次数
N_FOLDS = 5                # 最终模型训练的交叉验证折数

# Optuna 持久化存储配置
os.makedirs(SAVE_DIR, exist_ok=True) # 确保保存目录存在
storage_name = f"sqlite:///{os.path.join(SAVE_DIR, 'optuna_study.db')}"
study_name = "lgbm_f1_undersampling_v2" # 可为新实验更改此名称

# --- 1. 核心功能函数 ---
def translateCsvToDf(filepath, dtypes=None):
    """安全加载CSV文件"""
    return pd.read_csv(filepath, dtype=dtypes)

def feature_engineering(train_df, test_df):
    """特征工程主函数"""
    print("\n=== [1/5] 开始特征工程 ===")
    
    full_df = pd.concat([train_df.drop('is_new_did', axis=1, errors='ignore'), test_df], ignore_index=True)
    full_df['common_ts_dt'] = pd.to_datetime(full_df['common_ts'], unit='ms')
    
    did_stats = full_df.groupby('did').agg(
        mid_count=('mid', 'count'),
        eid_nunique=('eid', 'nunique'),
        ts_span=('common_ts_dt', lambda x: (x.max() - x.min()).total_seconds())
    ).reset_index()

    def process_chunk(df):
        """对数据块进行特征处理"""
        df['common_ts_dt'] = pd.to_datetime(df['common_ts'], unit='ms')
        df['hour'] = df['common_ts_dt'].dt.hour
        df['day_of_week'] = df['common_ts_dt'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        
        df['botId'] = np.nan
        df['pluginId'] = np.nan
        for idx, udmap_str in df['udmap'].items():
            if pd.isna(udmap_str) or udmap_str == '': continue
            try:
                udmap_dict = json.loads(udmap_str)
                if 'botId' in udmap_dict: df.loc[idx, 'botId'] = udmap_dict['botId']
                if 'pluginId' in udmap_dict: df.loc[idx, 'pluginId'] = udmap_dict['pluginId']
            except: continue
        
        df = df.merge(did_stats, on='did', how='left')
        
        for col in ['botId', 'pluginId']:
            if col in df.columns:
                df[col] = df[col].fillna(-1).astype('int32')
        return df

    train_processed = process_chunk(train_df.copy())
    test_processed = process_chunk(test_df.copy())
    
    print("✅ 特征工程完成。")
    return train_processed, test_processed

def find_best_f1_threshold(y_true, y_pred):
    """根据预测概率找到最佳F1阈值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# --- 2. 环境自检与数据加载 ---
print("代码开始执行...")
try:
    # ======================================================================
    #  核心修改点: 从下面的 lgb.train 调用中移除了已弃用的 verbose_eval 参数
    # ======================================================================
    lgb.train({'device': 'cuda'}, lgb.Dataset(np.random.rand(10,2), label=np.random.randint(0,2,10)), num_boost_round=1)
    GPU_AVAILABLE = True
    print("✅ GPU加速可用，将使用 'cuda' 设备。")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"⚠️ GPU不可用，将使用CPU。错误: {e}")

print("正在加载数据...")
OringinTrainDataUrl = os.path.join(DATA_DIR, "train_data/train.csv")
OringinTestDataUrl = os.path.join(DATA_DIR, "test_data/testA_data.csv")
train_df = translateCsvToDf(OringinTrainDataUrl)
test_df = translateCsvToDf(OringinTestDataUrl)

# --- 3. 执行特征工程与数据准备 ---
train_df, test_df = feature_engineering(train_df, test_df)

exclude_features = ['did', 'udmap', 'common_ts', 'common_ts_dt', 'is_new_did']
feature_cols = [col for col in train_df.columns if col not in exclude_features]
X = train_df[feature_cols]
y = train_df['is_new_did']
X_test = test_df[feature_cols]

print(f"模型将使用 {len(feature_cols)} 个特征。")

# --- 4. Optuna 自动调参 (可中断并续跑) ---
print(f"\n=== [2/5] 开始或继续Optuna调参（存入 {storage_name}） ===")

def objective(trial):
    """Optuna优化目标函数"""
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'verbosity': -1, 'random_state': 42, 
        'device': 'cuda' if GPU_AVAILABLE else 'cpu',
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
        
        model = lgb.train(params, lgb.Dataset(X_train_resampled, label=y_train_resampled),
                          valid_sets=[lgb.Dataset(X_val_fold, label=y_val_fold)],
                          num_boost_round=1500,
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        _, best_f1 = find_best_f1_threshold(y_val_fold, val_pred)
        cv_f1_scores.append(best_f1)
        
    return np.mean(cv_f1_scores)

# 创建或加载研究，实现断点续训
study = optuna.create_study(
    storage=storage_name,
    study_name=study_name,
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    load_if_exists=True 
)

completed_trials = len(study.trials)
print(f"Optuna研究 '{study_name}' 已有 {completed_trials} 次试验。")
if completed_trials < OPTUNA_TRIALS:
    remaining_trials = OPTUNA_TRIALS - completed_trials
    print(f"将继续优化，还需进行 {remaining_trials} 次试验。")
    study.optimize(objective, n_trials=remaining_trials, timeout=86400) # timeout设为24小时
else:
    print("所有优化试验已完成。")

print("\n✅ 优化完成！")
print(f"总试验次数: {len(study.trials)}")
print(f"最佳F1分数 (value): {study.best_value:.6f}")
best_params = study.best_params
print("最佳参数 (best_params):", best_params)

# --- 5. 最终模型训练与评估 ---
print(f"\n=== [3/5] 使用最佳参数进行 {N_FOLDS}-折交叉验证训练 ===")
best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 42, 'device': 'cuda' if GPU_AVAILABLE else 'cpu'})

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame(index=feature_cols)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLING_RATIO, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)
    
    model = lgb.train(best_params, lgb.Dataset(X_train_resampled, label=y_train_resampled),
                      valid_sets=[lgb.Dataset(X_val_fold, label=y_val_fold)],
                      valid_names=['valid'], num_boost_round=3000,
                      callbacks=[lgb.early_stopping(200, verbose=1000)])
    
    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    feature_importance[f'fold_{fold+1}'] = model.feature_importance(importance_type='gain')

print("\n✅ 交叉验证训练完成。")
print("\n=== [4/5] 开始模型评估 ===")
oof_best_threshold, oof_best_f1 = find_best_f1_threshold(y, oof_preds)
print(f"OOF 最佳F1分数: {oof_best_f1:.6f} (在阈值 {oof_best_threshold:.4f} 时取得)")
print("\nOOF 分类报告:")
print(classification_report(y, (oof_preds >= oof_best_threshold).astype(int), digits=4))

# --- 6. 结果保存 ---
print("\n=== [5/5] 开始生成并保存结果 ===")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({'did': test_df['did']})
submission['is_new_did'] = (test_preds >= oof_best_threshold).astype(int)
submission_filename = os.path.join(SAVE_DIR, f"submission_{timestamp}_f1_{oof_best_f1:.4f}.csv")
submission.to_csv(submission_filename, index=False)
print(f"📄 预测结果已保存至: {submission_filename}")

feature_importance['mean'] = feature_importance.mean(axis=1)
feature_importance.sort_values('mean', ascending=False, inplace=True)
feature_importance_filename = os.path.join(SAVE_DIR, f"feature_importance_{timestamp}.csv")
feature_importance.to_csv(feature_importance_filename)
print(f"📄 特征重要性已保存至: {feature_importance_filename}")

study_filename = os.path.join(SAVE_DIR, f"optuna_study_object_{timestamp}.pkl")
with open(study_filename, 'wb') as f:
    pickle.dump(study, f)
print(f"📄 Optuna研究对象已保存至: {study_filename}")

print("\n🎉 全部流程执行完毕！")