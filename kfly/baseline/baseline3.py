import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, f1_score, precision_recall_curve
from datetime import datetime
import warnings
import optuna
import os
warnings.filterwarnings('ignore')

def translateCsvToDf(filepath, dtypes=None):
    return pd.read_csv(filepath, dtype=dtypes)

# --- 1. GPU环境检查和数据加载 ---
print("代码开始执行...")

# 检查GPU是否可用
try:
    import lightgbm as lgb
    # 创建一个小的测试数据集来检查GPU
    test_X = np.random.rand(100, 5)
    test_y = np.random.randint(0, 2, 100)
    test_data = lgb.Dataset(test_X, label=test_y)
    
    test_params = {
        'objective': 'binary',
        'device': 'cuda',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbosity': -1
    }
    
    # 尝试使用GPU训练
    test_model = lgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
    print("✅ GPU加速可用，将使用GPU训练LightGBM")
    GPU_AVAILABLE = True
except Exception as e:
    print(f"⚠️  GPU不可用，将使用CPU训练: {e}")
    print("如需使用GPU，请确保：")
    print("1. 安装了支持GPU的LightGBM版本")
    print("2. 系统有可用的GPU和OpenCL驱动")
    print("3. 检查gpu_platform_id和gpu_device_id设置")
    GPU_AVAILABLE = False
dtype_mapping = {
    'mid': 'int32', 'eid': 'int32', 'device_brand': 'int16', 'ntt': 'int8',
    'operator': 'int8', 'common_country': 'int16', 'common_province': 'int16',
    'common_city': 'int16', 'appver': 'int16', 'channel': 'int16',
    'os_type': 'int8', 'is_new_did': 'int8'
}

OringinTrainDataUrl = r"/home/joker/new_csdiylearning2/kfly/data/train_data/train.csv"
OringinTestDataUrl = r"/home/joker/new_csdiylearning2/kfly/data/test_data/testA_data.csv"

print("加载训练数据...")
train_df = translateCsvToDf(OringinTrainDataUrl, dtypes=dtype_mapping)
print(f"训练数据形状: {train_df.shape}")

print("加载测试数据...")
test_df = translateCsvToDf(OringinTestDataUrl)
# 测试数据不包含目标变量，所以不在dtype_mapping中指定
print(f"测试数据形状: {test_df.shape}")

# --- 2. 数据探索 ---
print("\n=== 数据探索 ===")
print("训练数据基本信息:")
print(train_df.info())
print("\n目标变量分布:")
print(train_df['is_new_did'].value_counts())
print(f"新增用户比例: {train_df['is_new_did'].mean():.4f}")

print("\n缺失值情况:")
print("训练数据缺失值:")
print(train_df.isnull().sum())
print("\n测试数据缺失值:")
print(test_df.isnull().sum())

# --- 3. 特征工程 ---
def extract_udmap_features(df):
    """从udmap字段提取特征"""
    print("提取udmap特征...")
    
    # 初始化新特征
    df['botId'] = np.nan
    df['pluginId'] = np.nan
    df['udmap_is_valid'] = 0
    
    for idx, udmap_str in enumerate(df['udmap']):
        if pd.isna(udmap_str) or udmap_str == '':
            continue
        try:
            udmap_dict = json.loads(udmap_str)
            if 'botId' in udmap_dict:
                df.loc[idx, 'botId'] = udmap_dict['botId']
            if 'pluginId' in udmap_dict:
                df.loc[idx, 'pluginId'] = udmap_dict['pluginId']
            df.loc[idx, 'udmap_is_valid'] = 1
        except:
            continue
    
    return df

def create_time_features(df):
    """创建时间特征"""
    print("创建时间特征...")
    
    # 转换时间戳
    df['common_ts'] = pd.to_datetime(df['common_ts'], unit='ms')
    
    # 提取时间特征
    df['hour'] = df['common_ts'].dt.hour
    df['day_of_week'] = df['common_ts'].dt.dayofweek
    df['day_of_month'] = df['common_ts'].dt.day
    df['month'] = df['common_ts'].dt.month
    
    # 时间段特征
    df['time_period'] = pd.cut(df['hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=[0, 1, 2, 3], 
                              right=False).astype('int8')
    
    # 是否工作日
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    return df

def create_aggregation_features(df):
    """创建聚合特征"""
    print("创建聚合特征...")
    
    # 用户行为统计特征
    agg_features = []
    
    # 按did分组的统计特征
    did_stats = df.groupby('did').agg({
        'mid': ['count', 'nunique'],
        'eid': ['count', 'nunique'],
        'device_brand': 'nunique',
        'common_ts': ['min', 'max']
    }).reset_index()
    
    # 扁平化列名
    did_stats.columns = ['did'] + ['_'.join(col).strip() for col in did_stats.columns[1:]]
    
    # 计算时间跨度
    did_stats['ts_span'] = (did_stats['common_ts_max'] - did_stats['common_ts_min']).dt.total_seconds()
    
    # 合并到原数据
    df = df.merge(did_stats, on='did', how='left')
    
    return df

def feature_engineering(df, is_train=True):
    """特征工程主函数"""
    print(f"\n=== 特征工程 ({'训练' if is_train else '测试'}数据) ===")
    
    # 提取udmap特征
    df = extract_udmap_features(df)
    
    # 创建时间特征
    df = create_time_features(df)
    
    # 创建聚合特征
    df = create_aggregation_features(df)
    
    # 处理分类特征的缺失值
    categorical_cols = ['botId', 'pluginId']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype('int32')
    
    print(f"特征工程完成，数据形状: {df.shape}")
    return df

# 对训练和测试数据进行特征工程
train_df = feature_engineering(train_df, is_train=True)
test_df = feature_engineering(test_df, is_train=False)

# --- 4. 特征选择 ---
# 排除不需要的特征
exclude_features = ['did', 'udmap', 'common_ts', 'common_ts_min', 'common_ts_max']
if 'is_new_did' not in exclude_features:
    exclude_features.append('is_new_did')

feature_cols = [col for col in train_df.columns if col not in exclude_features]
print(f"\n使用特征数量: {len(feature_cols)}")
print("特征列表:", feature_cols)

# 准备训练数据
X = train_df[feature_cols]
y = train_df['is_new_did']
X_test = test_df[feature_cols]

print(f"训练特征矩阵形状: {X.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")

# --- 5. F1分数优化函数 ---
def find_best_threshold_f1(y_true, y_pred):
    """找到使F1分数最大的阈值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    return best_threshold, best_f1

# --- 6. Optuna贝叶斯优化调参（优化F1分数）---
print("\n=== Optuna贝叶斯优化调参（优化F1分数）===")

def objective_f1(trial):
    """Optuna优化目标函数 - 优化F1分数"""
    # 定义超参数搜索空间
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',  # 使用logloss作为训练指标
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'device': 'cuda',  # 使用GPU加速
        'gpu_platform_id': 0,  # GPU平台ID
        'gpu_device_id': 0,  # GPU设备ID
        
        # 为F1优化调整的超参数范围
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 300000),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        # 优化的超参数
        # 针对F1优化的额外参数
    }
    
    # 3折交叉验证评估参数
    skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_f1_scores = []
    
    for train_idx, val_idx in skf_opt.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # 静默训练
            ]
        )
        
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        
        # 找到最佳F1阈值并计算F1分数
        best_threshold, best_f1 = find_best_threshold_f1(y_val_fold, val_pred)
        cv_f1_scores.append(best_f1)
    
    return np.mean(cv_f1_scores)

# 开始优化
print("开始贝叶斯优化（优化F1分数）...")
study_f1 = optuna.create_study(direction='maximize', 
                              sampler=optuna.samplers.TPESampler(seed=42),
                              pruner=optuna.pruners.MedianPruner())

study_f1.optimize(objective_f1, n_trials=100, timeout=3600)  # 100次试验或1小时超时

print("优化完成！")
print(f"最佳F1分数: {study_f1.best_value:.6f}")
print("最佳参数:")
for key, value in study_f1.best_params.items():
    print(f"  {key}: {value}")

# 使用最佳参数
best_params_f1 = study_f1.best_params.copy()
best_params_f1.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'device': 'cuda' if GPU_AVAILABLE else 'cpu',
    'gpu_platform_id': 0 if GPU_AVAILABLE else None,
    'gpu_device_id': 0 if GPU_AVAILABLE else None,
    'num_threads': 0 if GPU_AVAILABLE else -1,
})

# 清理GPU模式下不支持的参数
if GPU_AVAILABLE:
    # 移除可能导致冲突的参数
    params_to_remove = ['n_jobs']
    for param in params_to_remove:
        if param in best_params_f1:
            del best_params_f1[param]

# --- 7. 使用最佳参数进行最终训练 ---
print("\n=== 使用最佳参数进行最终训练（F1优化）===")

# 5折交叉验证
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame()
fold_f1_scores = []
fold_thresholds = []

print("开始交叉验证训练...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    # 训练模型
    model = lgb.train(
        best_params_f1,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 预测验证集
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    # 预测测试集
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds += test_pred / n_fold
    
    # 保存特征重要性
    fold_importance = pd.DataFrame()
    fold_importance['feature'] = X.columns
    fold_importance['importance'] = model.feature_importance(importance_type='gain')
    fold_importance['fold'] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    # 计算该fold的最佳F1分数和阈值
    best_threshold, best_f1 = find_best_threshold_f1(y_val_fold, val_pred)
    fold_f1_scores.append(best_f1)
    fold_thresholds.append(best_threshold)
    
    # 验证集性能
    val_auc = roc_auc_score(y_val_fold, val_pred)
    print(f"Fold {fold + 1} AUC: {val_auc:.6f}")
    print(f"Fold {fold + 1} F1: {best_f1:.6f} (threshold: {best_threshold:.4f})")

# --- 8. 模型评估 ---
print("\n=== 模型评估 ===")

# 计算OOF AUC
oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF AUC: {oof_auc:.6f}")

# 计算OOF最佳F1阈值
oof_best_threshold, oof_best_f1 = find_best_threshold_f1(y, oof_preds)
print(f"OOF最佳F1分数: {oof_best_f1:.6f}")
print(f"OOF最佳阈值: {oof_best_threshold:.4f}")

# 计算各fold的平均F1和阈值
avg_fold_f1 = np.mean(fold_f1_scores)
avg_fold_threshold = np.mean(fold_thresholds)
print(f"各fold平均F1分数: {avg_fold_f1:.6f}")
print(f"各fold平均阈值: {avg_fold_threshold:.4f}")

# 使用OOF最佳阈值进行分类
oof_preds_binary = (oof_preds >= oof_best_threshold).astype(int)
oof_accuracy = accuracy_score(y, oof_preds_binary)
print(f"OOF准确率: {oof_accuracy:.6f}")

print("\n分类报告:")
print(classification_report(y, oof_preds_binary))

# --- 9. 特征重要性分析 ---
print("\n=== 特征重要性 ===")
feature_importance_agg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
print("Top 20 重要特征:")
print(feature_importance_agg.head(20))

# --- 10. 生成预测结果 ---
print("\n=== 生成预测结果 ===")

# 创建保存目录
save_dir = "/home/joker/new_csdiylearning2/kfly/data/baseline3"
os.makedirs(save_dir, exist_ok=True)

# 准备提交文件
submission = pd.DataFrame()
submission['did'] = test_df['did']
submission['is_new_did'] = test_preds

# 使用OOF最佳F1阈值进行二分类
submission['is_new_did_binary'] = (test_preds >= oof_best_threshold).astype(int)

# 也可以使用各fold平均阈值
submission['is_new_did_binary_avg'] = (test_preds >= avg_fold_threshold).astype(int)

print("预测结果统计:")
print(f"预测概率均值: {test_preds.mean():.4f}")
print(f"预测概率标准差: {test_preds.std():.4f}")
print(f"预测为新用户的比例（OOF阈值）: {submission['is_new_did_binary'].mean():.4f}")
print(f"预测为新用户的比例（平均阈值）: {submission['is_new_did_binary_avg'].mean():.4f}")

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = os.path.join(save_dir, f"submission_lgb_f1_optuna_{timestamp}.csv")
binary_filename = os.path.join(save_dir, f"submission_lgb_f1_binary_optuna_{timestamp}.csv")
binary_avg_filename = os.path.join(save_dir, f"submission_lgb_f1_binary_avg_optuna_{timestamp}.csv")

# 保存概率预测结果
submission[['did', 'is_new_did']].to_csv(submission_filename, index=False)
print(f"预测结果已保存至: {submission_filename}")

# 保存F1优化的二分类结果（OOF阈值）
submission[['did', 'is_new_did_binary']].rename(columns={'is_new_did_binary': 'is_new_did'}).to_csv(binary_filename, index=False)
print(f"F1优化二分类结果（OOF阈值）已保存至: {binary_filename}")

# 保存F1优化的二分类结果（平均阈值）
submission[['did', 'is_new_did_binary_avg']].rename(columns={'is_new_did_binary_avg': 'is_new_did'}).to_csv(binary_avg_filename, index=False)
print(f"F1优化二分类结果（平均阈值）已保存至: {binary_avg_filename}")

# 保存优化历史和最佳参数
optuna_results_f1 = {
    'best_f1': study_f1.best_value,
    'best_params': study_f1.best_params,
    'final_oof_auc': oof_auc,
    'final_oof_f1': oof_best_f1,
    'oof_best_threshold': oof_best_threshold,
    'fold_avg_f1': avg_fold_f1,
    'fold_avg_threshold': avg_fold_threshold,
    'optimization_trials': len(study_f1.trials)
}

import pickle
optuna_filename = os.path.join(save_dir, f"optuna_f1_study_{timestamp}.pkl")
with open(optuna_filename, 'wb') as f:
    pickle.dump(study_f1, f)
print(f"Optuna F1优化研究结果已保存至: {optuna_filename}")

# 保存特征重要性
feature_importance_agg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
feature_importance_filename = os.path.join(save_dir, f"feature_importance_f1_{timestamp}.csv")
feature_importance_agg.to_csv(feature_importance_filename)
print(f"特征重要性已保存至: {feature_importance_filename}")

# 保存阈值信息
threshold_info = pd.DataFrame({
    'fold': list(range(1, n_fold + 1)) + ['OOF', 'Average'],
    'f1_score': fold_f1_scores + [oof_best_f1, avg_fold_f1],
    'threshold': fold_thresholds + [oof_best_threshold, avg_fold_threshold]
})
threshold_filename = os.path.join(save_dir, f"f1_thresholds_{timestamp}.csv")
threshold_info.to_csv(threshold_filename, index=False)
print(f"F1阈值信息已保存至: {threshold_filename}")

print("\n=== F1优化基线模型完成 ===")
print(f"使用设备: {'GPU' if GPU_AVAILABLE else 'CPU'}")
print(f"Optuna最佳验证F1: {study_f1.best_value:.6f}")
print(f"最终模型OOF AUC: {oof_auc:.6f}")
print(f"最终模型OOF F1: {oof_best_f1:.6f}")
print(f"最终模型最佳阈值: {oof_best_threshold:.4f}")
print(f"使用了 {len(feature_cols)} 个特征")
print(f"优化试验次数: {len(study_f1.trials)}")
print("\n针对F1优化的改进:")
print("1. 优化目标从AUC改为F1分数")
print("2. 增加了scale_pos_weight和is_unbalance参数优化")
print("3. 为每个fold计算最佳F1阈值")
print("4. 提供了多种阈值选择策略")
print("5. 保存了详细的F1和阈值信息")
if GPU_AVAILABLE:
    print("6. 🚀 使用GPU加速训练，显著提升训练速度")
    print("7. 🎯 GPU特定参数优化（gpu_use_dp等）")
print("\n建议后续优化方向:")
print("1. 特征工程：更多时间窗口统计特征、用户行为序列特征")
print("2. 模型融合：XGBoost、CatBoost等模型集成")
print("3. 类别平衡：SMOTE、随机欠采样等技术")
print("4. 阈值策略：动态阈值、基于验证集的阈值选择")
print("5. 集成学习：多个F1优化模型的ensemble")
if GPU_AVAILABLE:
    print("6. GPU优化：调整gpu_use_dp、增加GPU内存使用效率")
    print("7. 并行训练：多GPU训练、分布式训练")
else:
    print("6. GPU加速：安装GPU版本LightGBM获得更快训练速度")