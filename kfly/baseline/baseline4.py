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
import gc # 匯入記憶體回收模組

warnings.filterwarnings('ignore')

#==============================================================================
#  日誌與配置 (Logging and Configuration)
#==============================================================================
# 在腳本開頭設定全局變數，方便調整
# Optuna 設定
N_TRIALS = 50  # Optuna 試驗次數，可根據時間預算調整
TIMEOUT = 7200 # Optuna 最大運行時間（秒），例如 2 小時

# 偽標籤設定
USE_PSEUDO_LABELING = True # 是否啟用偽標籤增強訓練
PL_THRESHOLD_HIGH = 0.98   # 偽標籤為 1 的置信度閾值
PL_THRESHOLD_LOW = 0.02    # 偽標籤為 0 的置信度閾值

# 交叉驗證折數
N_FOLDS_OPTUNA = 3 # Optuna 內部使用的折數
N_FOLDS_TRAIN = 5  # 最終訓練使用的折數

#==============================================================================
# --- 1. 環境檢查、資料加載與輔助函數 ---
#==============================================================================

def translateCsvToDf(filepath, dtypes=None):
    return pd.read_csv(filepath, dtype=dtypes)

def find_best_threshold_f1(y_true, y_pred):
    """找到使F1分數最大的閾值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    # 閾值數組比f1分數多一個元素，處理邊界情況
    f1_scores = f1_scores[:-1]
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    return best_threshold, best_f1

print("代碼開始執行...")

# 檢查GPU是否可用
try:
    test_X = np.random.rand(10, 2)
    test_y = np.random.randint(0, 2, 10)
    test_data = lgb.Dataset(test_X, label=test_y)
    test_params = {'objective': 'binary', 'device': 'cuda', 'verbosity': -1}
    # [BUG修復] verbose_eval 不是 train 的有效參數，應使用回呼函數
    lgb.train(test_params, test_data, num_boost_round=1, callbacks=[lgb.log_evaluation(period=0)])
    print("✅ CUDA GPU 加速可用，將使用 GPU 訓練 LightGBM")
    GPU_AVAILABLE = True
except Exception as e:
    print(f"⚠️ GPU 不可用，將使用 CPU 訓練: {e}")
    GPU_AVAILABLE = False

dtype_mapping = {
    'mid': 'int32', 'eid': 'int32', 'device_brand': 'category', 'ntt': 'category',
    'operator': 'category', 'common_country': 'category', 'common_province': 'category',
    'common_city': 'category', 'appver': 'category', 'channel': 'category',
    'os_type': 'category', 'is_new_did': 'int8'
}

OringinTrainDataUrl = r"/home/joker/new_csdiylearning2/kfly/data/train_data/train.csv"
OringinTestDataUrl = r"/home/joker/new_csdiylearning2/kfly/data/test_data/testA_data.csv"

print("加載訓練與測試數據...")
train_df = translateCsvToDf(OringinTrainDataUrl, dtypes=dtype_mapping)
test_df = translateCsvToDf(OringinTestDataUrl, dtypes={k: v for k, v in dtype_mapping.items() if k != 'is_new_did'})
train_size = len(train_df)

print(f"訓練數據形狀: {train_df.shape}")
print(f"測試數據形狀: {test_df.shape}")
print("\n目標變數分佈:")
print(train_df['is_new_did'].value_counts(normalize=True))

#==============================================================================
# --- 2. 高級特徵工程 ---
#==============================================================================
print("\n=== 高級特徵工程 ===")

# [新策略] 合併處理，簡化流程
# 在測試集中新增 is_new_did 欄位，方便合併
test_df['is_new_did'] = -1 
data_df = pd.concat([train_df, test_df], ignore_index=True)
del train_df, test_df
gc.collect()

# 1. 時間特徵
print("創建時間特徵...")
data_df['common_ts'] = pd.to_datetime(data_df['common_ts'], unit='ms')
data_df['hour'] = data_df['common_ts'].dt.hour
data_df['day_of_week'] = data_df['common_ts'].dt.dayofweek
data_df['day_of_month'] = data_df['common_ts'].dt.day
data_df['is_weekend'] = (data_df['day_of_week'] >= 5).astype('int8')

# [新特徵] 時間差特徵 (Time Delta)
print("創建時間差特徵...")
data_df = data_df.sort_values(by=['did', 'common_ts'])
data_df['prev_ts'] = data_df.groupby('did')['common_ts'].shift(1)
data_df['time_delta_seconds'] = (data_df['common_ts'] - data_df['prev_ts']).dt.total_seconds().fillna(0)

# 2. udmap 特徵
print("提取udmap特徵...")
data_df['botId'] = np.nan
data_df['pluginId'] = np.nan
# 使用 apply 以提升效率
def parse_udmap(udmap_str):
    if pd.isna(udmap_str) or udmap_str == '':
        return np.nan, np.nan
    try:
        udmap_dict = json.loads(udmap_str)
        bot_id = udmap_dict.get('botId', np.nan)
        plugin_id = udmap_dict.get('pluginId', np.nan)
        return bot_id, plugin_id
    except:
        return np.nan, np.nan
udmap_parsed = data_df['udmap'].apply(parse_udmap)
data_df['botId'] = udmap_parsed.apply(lambda x: x[0])
data_df['pluginId'] = udmap_parsed.apply(lambda x: x[1])
data_df['botId'] = data_df['botId'].fillna(-1).astype('int32')
data_df['pluginId'] = data_df['pluginId'].fillna(-1).astype('int32')

# 3. 聚合特徵
print("創建聚合特徵...")
# 為了避免在合併後計算聚合特徵時洩漏未來資訊，先按 did 分組
# 計算每個 did 的首次和末次時間
did_time_stats = data_df.groupby('did')['common_ts'].agg(['min', 'max']).reset_index()
did_time_stats.columns = ['did', 'did_ts_min', 'did_ts_max']
did_time_stats['did_ts_span_seconds'] = (did_time_stats['did_ts_max'] - did_time_stats['did_ts_min']).dt.total_seconds()
data_df = data_df.merge(did_time_stats, on='did', how='left')

# [新特徵] 行為序列特徵
# 計算每條記錄距離使用者首次行為的時間
data_df['time_since_first_event'] = (data_df['common_ts'] - data_df['did_ts_min']).dt.total_seconds()

# 其他聚合特徵
agg_dict = {
    'eid': ['count', 'nunique'],
    'mid': ['nunique'],
    'time_delta_seconds': ['mean', 'std', 'max', 'sum']
}
did_agg_stats = data_df.groupby('did').agg(agg_dict).reset_index()
did_agg_stats.columns = ['did'] + ['did_' + '_'.join(col).strip() for col in did_agg_stats.columns[1:]]
data_df = data_df.merge(did_agg_stats, on='did', how='left')

# [新特徵] 交叉組合與頻率編碼
print("創建交叉組合與頻率特徵...")
cat_cols_for_comb = ['mid', 'eid', 'device_brand', 'os_type', 'channel']
for i in range(len(cat_cols_for_comb)):
    for j in range(i + 1, len(cat_cols_for_comb)):
        col1 = cat_cols_for_comb[i]
        col2 = cat_cols_for_comb[j]
        new_col_name = f'{col1}_{col2}_comb'
        data_df[new_col_name] = data_df[col1].astype(str) + '_' + data_df[col2].astype(str)
        # 頻率編碼
        freq_map = data_df[new_col_name].value_counts().to_dict()
        data_df[f'{new_col_name}_freq'] = data_df[new_col_name].map(freq_map)
        data_df = data_df.drop(new_col_name, axis=1) # 刪除臨時的組合欄位

# ===== 修復：確保所有分類變量都被正確轉換 =====
print("轉換所有分類變量為數值類型...")

# 1. 先處理 category 類型
category_cols = data_df.select_dtypes(include='category').columns.tolist()
print(f"Category 類型列: {category_cols}")
for col in category_cols:
    print(f"轉換 category 列: {col}")
    data_df[col] = data_df[col].cat.codes

# 2. 處理 object 類型（字符串）
object_cols = data_df.select_dtypes(include='object').columns.tolist()
# 排除不需要转换的列（如时间戳等）
exclude_from_encoding = ['udmap', 'common_ts', 'prev_ts', 'did_ts_min', 'did_ts_max']
object_cols = [col for col in object_cols if col not in exclude_from_encoding]

print(f"需要轉換的 object 列: {object_cols}")

for col in object_cols:
    print(f"轉換 object 列: {col}")
    # 使用 LabelEncoder 的逻辑
    unique_values = data_df[col].dropna().unique()
    value_to_code = {val: idx for idx, val in enumerate(unique_values)}
    value_to_code[None] = -1  # 处理 NaN
    # 对于 numpy.nan 也要处理
    data_df[col] = data_df[col].fillna('__NAN__')  # 先填充 NaN
    unique_values_with_nan = data_df[col].unique()
    value_to_code = {val: idx if val != '__NAN__' else -1 for idx, val in enumerate(unique_values_with_nan)}
    
    data_df[col] = data_df[col].map(value_to_code).astype('int32')

# 3. 验证数据类型
print("\n驗證數據類型...")
problematic_cols = []
for col in data_df.columns:
    dtype = data_df[col].dtype
    if dtype == 'object':
        problematic_cols.append(col)
        print(f"  ⚠️  警告: {col} 仍然是 object 類型!")
        print(f"  樣本值: {data_df[col].head().tolist()}")

if problematic_cols:
    print(f"\n仍有 {len(problematic_cols)} 個 object 類型列需要處理")
    # 强制转换剩余的 object 列
    for col in problematic_cols:
        if col not in exclude_from_encoding:
            print(f"強制轉換 {col}")
            data_df[col] = pd.factorize(data_df[col])[0]
else:
    print("✅ 所有分類變量已成功轉換為數值類型")

print(f"特徵工程完成，數據形狀: {data_df.shape}")
gc.collect()


#==============================================================================
# --- 3. 訓練主流程 ---
#==============================================================================

def train_main(X, y, X_test, stage='initial'):
    """主訓練函數，封裝了交叉驗證、模型訓練和預測"""
    print(f"\n=== 開始第 {stage} 階段訓練 ===")
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame()
    
    # [修復] 目標編碼的欄位列表
    target_encode_cols = ['mid', 'eid', 'device_brand', 'os_type', 'channel', 'common_city', 'appver']
    # 确保目标编码的列在特征中存在
    target_encode_cols = [col for col in target_encode_cols if col in X.columns]

    skf = StratifiedKFold(n_splits=N_FOLDS_TRAIN, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS_TRAIN} ---")
        
        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # [修復] 在 Fold 內部進行安全的目標編碼
        print("進行目標編碼...")
        X_test_fold = X_test.copy()
        
        for col in target_encode_cols:
            if col in X_train_fold.columns:
                # [修復] 創建包含特徵和目標的臨時DataFrame來進行目標編碼
                temp_train_df = X_train_fold[[col]].copy()
                temp_train_df['target'] = y_train_fold.values
                
                # 計算訓練折內的目標均值
                target_mean = temp_train_df.groupby(col)['target'].mean()
                
                # 應用到訓練折和驗證折
                X_train_fold[f'{col}_target_enc'] = X_train_fold[col].map(target_mean)
                X_val_fold[f'{col}_target_enc'] = X_val_fold[col].map(target_mean)
                # 處理驗證集中可能出現的新類別，用全域均值填充
                X_val_fold[f'{col}_target_enc'].fillna(y_train_fold.mean(), inplace=True)
                
                # 應用到測試集（每次都重新計算，最後取平均）
                X_test_fold[f'{col}_target_enc_{fold}'] = X_test[col].map(target_mean)
                X_test_fold[f'{col}_target_enc_{fold}'].fillna(y_train_fold.mean(), inplace=True)

        # 準備訓練資料
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        model = lgb.train(
            best_params_f1,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=3000, # 增加輪數，配合早停
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=True),
                lgb.log_evaluation(period=200)
            ]
        )
        
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        # 测试集预测时要注意特征对齐
        test_features = [col for col in X_train_fold.columns if not col.endswith(f'_{fold}')]
        X_test_pred = X_test_fold[X_train_fold.columns]
        test_pred = model.predict(X_test_pred, num_iteration=model.best_iteration)
        test_preds += test_pred / skf.n_splits
        
        # 記錄特徵重要性
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train_fold.columns
        fold_importance_df["importance"] = model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # 将这一折的目标编码特征添加到测试集
        for col in target_encode_cols:
            if col in X_train_fold.columns:
                if f'{col}_target_enc_{fold}' not in X_test.columns:
                    X_test[f'{col}_target_enc_{fold}'] = X_test_fold[f'{col}_target_enc_{fold}']

    # 處理測試集的目標編碼特徵（取各折的平均值）
    for col in target_encode_cols:
        if col in X.columns:
            te_cols_in_test = [f'{col}_target_enc_{f}' for f in range(N_FOLDS_TRAIN) if f'{col}_target_enc_{f}' in X_test.columns]
            if te_cols_in_test:
                X_test[f'{col}_target_enc'] = X_test[te_cols_in_test].mean(axis=1)
                X_test.drop(columns=te_cols_in_test, inplace=True)

    # 評估 OOF 預測結果
    oof_best_threshold, oof_best_f1 = find_best_threshold_f1(y, oof_preds)
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n=== {stage} 階段 OOF 評估 ===")
    print(f"OOF AUC: {oof_auc:.6f}")
    print(f"OOF F1: {oof_best_f1:.6f} (最佳閾值: {oof_best_threshold:.4f})")
    
    return oof_preds, test_preds, feature_importance_df, oof_best_threshold


#==============================================================================
# --- 4. Optuna 超參數優化 ---
#==============================================================================
print("\n=== Optuna 貝葉斯優化調參 ===")

# 分割回原始訓練集和測試集以進行後續步驟
train_df = data_df[data_df['is_new_did'] != -1].copy()
test_df = data_df[data_df['is_new_did'] == -1].copy()
del data_df
gc.collect()

# 定義特徵列
# 排除 ID、原始時間戳和目標變數
exclude_features = ['did', 'udmap', 'common_ts', 'is_new_did', 'prev_ts', 'did_ts_min', 'did_ts_max']
feature_cols = [col for col in train_df.columns if col not in exclude_features]

X = train_df[feature_cols]
y = train_df['is_new_did']
X_test = test_df[feature_cols]

print(f"最終使用特徵數量: {len(feature_cols)}")

# 最后再检查一次数据类型
print("最終數據類型檢查...")
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"發現 object 類型: {col}, 強制轉換...")
        X[col] = pd.factorize(X[col])[0]
        X_test[col] = pd.factorize(X_test[col])[0]

def objective_f1(trial):
    # 更精細的超參數搜索空間
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'device': 'cuda' if GPU_AVAILABLE else 'cpu',
        'n_jobs': -1,
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-6, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-6, 10.0, log=True),
    }

    skf_opt = StratifiedKFold(n_splits=N_FOLDS_OPTUNA, shuffle=True, random_state=1337)
    cv_f1_scores = []
    
    for train_idx, val_idx in skf_opt.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.train(
            params,
            lgb.Dataset(X_train_fold, label=y_train_fold),
            valid_sets=[lgb.Dataset(X_val_fold, label=y_val_fold)],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)]
        )
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        _, best_f1 = find_best_threshold_f1(y_val_fold, val_pred)
        cv_f1_scores.append(best_f1)
        
    return np.mean(cv_f1_scores)

print(f"開始貝葉斯優化，共 {N_TRIALS} 次試驗...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_f1, n_trials=N_TRIALS, timeout=TIMEOUT)

print("優化完成！")
print(f"最佳驗證 F1 分數: {study.best_value:.6f}")
best_params_f1 = study.best_params
print("最佳參數:", best_params_f1)

# 更新固定參數
best_params_f1.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'device': 'cuda' if GPU_AVAILABLE else 'cpu',
})

#==============================================================================
# --- 5. 最終模型訓練與偽標籤 ---
#==============================================================================

# 初始訓練
oof_preds_initial, test_preds_initial, importance_df_initial, threshold_initial = train_main(X, y, X_test, stage='initial')

if USE_PSEUDO_LABELING:
    print("\n=== 開始偽標籤增強訓練階段 ===")
    
    # 創建偽標籤
    test_df['pseudo_label'] = -1
    high_conf_idx = np.where(test_preds_initial >= PL_THRESHOLD_HIGH)[0]
    low_conf_idx = np.where(test_preds_initial <= PL_THRESHOLD_LOW)[0]
    
    test_df.loc[high_conf_idx, 'pseudo_label'] = 1
    test_df.loc[low_conf_idx, 'pseudo_label'] = 0
    
    pseudo_labeled_df = test_df[test_df['pseudo_label'] != -1]
    print(f"找到 {len(pseudo_labeled_df)} 個高置信度偽標籤樣本。")
    
    if len(pseudo_labeled_df) > 0:
        # 合併偽標籤資料
        X_pseudo = pseudo_labeled_df[feature_cols]
        y_pseudo = pseudo_labeled_df['pseudo_label']
        
        X_augmented = pd.concat([X, X_pseudo], ignore_index=True)
        y_augmented = pd.concat([y, y_pseudo], ignore_index=True)
        
        print(f"增強後訓練集大小: {len(X_augmented)}")
        
        # 使用增強後的資料集進行最終訓練
        oof_preds_final, test_preds_final, importance_df_final, threshold_final = train_main(
            X_augmented, y_augmented, X_test, stage='pseudo_labeling'
        )
    else:
        print("未找到足夠的偽標籤樣本，跳過增強訓練。")
        oof_preds_final, test_preds_final, importance_df_final, threshold_final = \
            oof_preds_initial, test_preds_initial, importance_df_initial, threshold_initial
else:
    print("未啟用偽標籤，使用初始訓練結果。")
    oof_preds_final, test_preds_final, importance_df_final, threshold_final = \
        oof_preds_initial, test_preds_initial, importance_df_initial, threshold_initial


#==============================================================================
# --- 6. 結果儲存與分析 ---
#==============================================================================
print("\n=== 生成並儲存最終結果 ===")

# 創建保存目錄
save_dir = "/home/joker/new_csdiylearning2/kfly/data/baseline4"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 準備提交文件
submission = pd.DataFrame({'did': test_df['did'], 'is_new_did_prob': test_preds_final})
submission['is_new_did'] = (submission['is_new_did_prob'] >= threshold_final).astype(int)

# 儲存二分類結果
binary_filename = os.path.join(save_dir, f"submission_{timestamp}.csv")
submission[['did', 'is_new_did']].to_csv(binary_filename, index=False)
print(f"提交檔案已儲存至: {binary_filename}")
print("預測為新使用者的比例:", submission['is_new_did'].mean())

# 儲存概率結果
prob_filename = os.path.join(save_dir, f"submission_prob_{timestamp}.csv")
submission[['did', 'is_new_did_prob']].to_csv(prob_filename, index=False)
print(f"概率預測檔案已儲存至: {prob_filename}")

# 儲存特徵重要性
feature_importance_agg = importance_df_final.groupby('feature')['importance'].mean().sort_values(ascending=False)
feature_importance_filename = os.path.join(save_dir, f"feature_importance_{timestamp}.csv")
feature_importance_agg.to_csv(feature_importance_filename)
print(f"特徵重要性已儲存至: {feature_importance_filename}")

print("\n=== 所有流程執行完畢 ===")