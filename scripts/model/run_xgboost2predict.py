'''
name: run_xgboost2predict.py
usage: use XGBoost to train and predict
input: /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy
       /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv
output: /root/autodl-tmp/project/mlpj/results/logs/xgboost_2025_standard_results.npz
authors/date: Q/2025.12.16
'''

import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ================= 0. 标准化协议：随机种子 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Protocol] Random Seed set to {seed}")

set_seed(42)

# ================= 1. 配置区域 =================
# 输入路径
data_npy = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy"
data_csv = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv"

# 输出路径
save_dir = "/root/autodl-tmp/project/mlpj/results/logs"
os.makedirs(save_dir, exist_ok=True)
output_npz = os.path.join(save_dir, "xgboost_2025_standard_results.npz")
output_img = "xgboost_2025_standard.png"

# 标准化参数
context_len = 512  # 输入过去 512 小时
pred_len = 24      # 预测未来 24 小时
stride = 24        # 滚动步长 (测试时)

# ================= 2. 数据加载与标准化预处理 =================
print("1. [Protocol] Loading & Preprocessing Data...")
if not os.path.exists(data_npy) or not os.path.exists(data_csv):
    print("❌ Data not found.")
    exit()

# 加载原始数据
data_values = np.load(data_npy) # (N,)
if data_values.ndim == 1:
    data_values = data_values.reshape(-1, 1)

df = pd.read_csv(data_csv)
df['time'] = pd.to_datetime(df['time'])

# --- 辅助特征：时间 Embedding (XGBoost 需要这个来感知季节) ---
# 这不违反"同一套数据集"原则，因为这是从时间戳中提取的固有信息
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['dayofyear'] = df['time'].dt.dayofyear
time_feats = df[['hour', 'month', 'dayofyear']].values

# --- 严格划分训练集与测试集 ---
split_date = pd.Timestamp("2025-01-01 00:00:00")
train_mask = df['time'] < split_date

# 训练集: 2010-2024
train_data_raw = data_values[train_mask]
train_time_raw = time_feats[train_mask]

# 测试起点索引
test_start_idx = np.sum(train_mask)
print(f"   -> Train Data End: {df['time'].iloc[test_start_idx-1]}")
print(f"   -> Test Start:     {df['time'].iloc[test_start_idx]}")

# --- 统一预处理: MinMaxScaler (0~1) ---
# 规则: 只能用训练集(2010-2024) fit
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data_raw)

# 转换全量数据
data_scaled = scaler.transform(data_values).flatten()
print(f"   -> [Protocol] Scaler fitted on Train set (2010-2024).")

# ================= 3. 构造训练样本 =================
# 我们需要构造 (X, y) 对来训练 XGBoost
# X: [Lag_1 ... Lag_512, Hour, Month, Day]
# y: [Future_1 ... Future_24]

def create_dataset(data_seq, time_seq, context_len, pred_len, stride=1):
    X, y = [], []
    # 训练时 stride 可以小一点(如1或12)以增加样本量
    # 这里设为 12 以兼顾速度和精度
    for i in range(0, len(data_seq) - context_len - pred_len + 1, stride):
        # 1. 历史气温特征 (512维)
        lags = data_seq[i : i + context_len]
        
        # 2. 预测时刻的时间特征 (3维) - 取预测窗口的第一个时间点
        # 告诉模型"我们要预测什么时候的气温"
        curr_time = time_seq[i + context_len] 
        
        # 合并
        feature_vector = np.concatenate([lags, curr_time])
        
        # 3. 目标 (24维)
        target = data_seq[i + context_len : i + context_len + pred_len]
        
        X.append(feature_vector)
        y.append(target)
    return np.array(X), np.array(y)

print("2. Constructing Training Samples...")
X_train, y_train = create_dataset(
    data_scaled[train_mask], 
    train_time_raw, 
    context_len, 
    pred_len, 
    stride=12 # 训练采样步长
)
print(f"   -> Training Samples: {X_train.shape}")

# ================= 4. 训练模型 =================
print("3. Training XGBoost (MultiOutput)...")
# 使用 GPU 加速
xgb_params = {
    'n_estimators': 800,
    'learning_rate': 0.05,
    'max_depth': 8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'  
}

# MultiOutputRegressor: 一次性预测24个点，对应 Sundial 的 generate(24)
model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
model.fit(X_train, y_train)
print("   ✅ Training Complete.")

# ================= 5. 滚动预测 (Standard Rolling) =================
print(f"4. [Protocol] Rolling Forecast (Stride={stride})...")

all_preds = []
all_trues = []
timestamps = []

# 迭代器: 从 2025-01-01 开始
idx_iter = range(test_start_idx, len(data_values) - pred_len, stride)

# 准备测试数据 (全量归一化后的)
test_data_scaled = data_scaled
test_time_feats = time_feats

for current_idx in tqdm(idx_iter, desc="Inferencing"):
    # A. 准备输入
    # 历史归一化数据
    history_norm = test_data_scaled[current_idx - context_len : current_idx]
    # 当前预测点的时间特征
    curr_time_feat = test_time_feats[current_idx]
    
    # 拼接输入向量
    input_vec = np.concatenate([history_norm, curr_time_feat]).reshape(1, -1)
    
    # B. 推理 (得到归一化后的 24 小时预测)
    pred_norm = model.predict(input_vec).flatten()
    
    # C. 获取真实值
    true_norm = test_data_scaled[current_idx : current_idx + pred_len]
    
    # D. 反归一化
    pred_real = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    true_real = scaler.inverse_transform(true_norm.reshape(-1, 1)).flatten()
    
    all_preds.extend(pred_real)
    all_trues.extend(true_real)
    timestamps.extend(df['time'].iloc[current_idx : current_idx + pred_len])

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

# ================= 6. 计算指标 (Standard Metrics) =================
mse = mean_squared_error(all_trues, all_preds)
mae = mean_absolute_error(all_trues, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_trues, all_preds)

print("\n" + "="*40)
print(f"📊 XGBoost 2025 Evaluation Report:")
print(f"   MAE  : {mae:.4f} °C")
print(f"   RMSE : {rmse:.4f} °C")
print(f"   MSE  : {mse:.4f}")
print(f"   R²   : {r2:.4f}")
print("="*40)

# ================= 7. 保存结果 =================
np.savez(output_npz, preds=all_preds, trues=all_trues, mse=mse, mae=mae, rmse=rmse, r2=r2)
print(f"💾 Results saved to {output_npz}")
