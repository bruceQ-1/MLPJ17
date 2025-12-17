'''
name: run_xgboost_gap_2025.py
usage: Train XGBoost on 2010-2014, Predict 2025 
       使用七个变量训练xgboost
input: /root/autodl-tmp/project/mlpj/data/processed/train_2010_2014_rich.npy
       /root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.npy
output: /root/autodl-tmp/project/mlpj/results/logs/xgboost_gap_2025_results.npz
        /root/autodl-tmp/project/mlpj/results/figures/xgboost_gap_2025.png
'''

import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ================= 0. 标准化协议：随机种子 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 [Protocol] Random Seed set to {seed}")

set_seed(42)

# ================= 1. 配置区域 =================
# 路径 
train_npy = "/root/autodl-tmp/project/mlpj/data/processed/train_2010_2014_rich.npy"
test_npy  = "/root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.npy"
test_csv  = "/root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.csv"

# 保存路径
save_dir = "/root/autodl-tmp/project/mlpj/results/logs"
os.makedirs(save_dir, exist_ok=True)
output_npz = os.path.join(save_dir, "xgboost_gap_2025_results.npz")
output_img = "/root/autodl-tmp/project/mlpj/results/figures/xgboost_gap_2025.png"

# 参数
context_len = 512
pred_len = 24
input_dims = 7 # 7个气象特征

# ================= 2. 数据加载与预处理 =================
print("1. Loading Data...")
if not os.path.exists(train_npy) or not os.path.exists(test_npy):
    print("❌ 找不到数据文件，请检查 processed 文件夹")
    exit()

train_data = np.load(train_npy) # (N_train, 7)
test_data  = np.load(test_npy)  # (N_test, 7)

print(f"   Train Shape: {train_data.shape} (2010-2014)")
print(f"   Test Shape:  {test_data.shape} (2025)")

# 归一化 (关键: 只在训练集上 Fit)
scaler = MinMaxScaler((0, 1))
scaler.fit(train_data)

train_scaled = scaler.transform(train_data)
test_scaled  = scaler.transform(test_data)

# ================= 3. 构造 Flatten 数据集 =================
# XGBoost 无法直接吃 (512, 7) 的 3D 数据，需要展平为 (512*7) 的 1D 向量
def create_xgb_dataset(data, context_len, pred_len, stride=1):
    X, y = [], []
    for i in range(0, len(data) - context_len - pred_len + 1, stride):
        # Input: 过去 512 小时的 7 个特征 -> Flatten
        # Shape: (512 * 7,) = (3584,)
        feature_vec = data[i : i+context_len].flatten()
        
        # Output: 未来 24 小时的气温 (第0列)
        target = data[i+context_len : i+context_len+pred_len, 0]
        
        X.append(feature_vec)
        y.append(target)
    return np.array(X), np.array(y)

print("2. Constructing Training Samples...")
# 训练时 stride=12 (降采样以节省内存和时间，但保证样本量足够)
# 如果内存只有 35GB，3584维特征 x 数万样本是可以吃下的
X_train, y_train = create_xgb_dataset(train_scaled, context_len, pred_len, stride=12)

print(f"   -> Feature Dims: {X_train.shape[1]} (512 * 7)")
print(f"   -> Samples: {X_train.shape[0]}")

# ================= 4. 训练 XGBoost =================
print("3. Training XGBoost (MultiOutput)...")

# 配置 GPU
xgb_params = {
    'n_estimators': 800,
    'learning_rate': 0.05,
    'max_depth': 8,             # 树深一点，因为特征维度很高
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'tree_method': 'hist',      # 必须用 hist 模式加速
    'device': 'cuda',           # 使用 GPU
    'colsample_bytree': 0.6     # 每次只用 60% 的特征，防止过拟合
}

model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
model.fit(X_train, y_train)
print("   ✅ Training Complete.")

# ================= 5. 预测 2025 (Gap Test) =================
print("4. Predicting 2025 (Gap Test)...")

all_preds = []
all_trues = []
timestamps = []

# 生成 2025 时间轴
if os.path.exists(test_csv):
    df_test = pd.read_csv(test_csv)
    time_index = pd.to_datetime(df_test['time'])
else:
    time_index = pd.date_range(start="2025-01-01", periods=len(test_data), freq='H')

# 滚动预测
# 步长 24
idx_iter = range(0, len(test_scaled) - context_len - pred_len, 24)

# 准备测试数据 (预先构建好可以加速，但为了省内存还是循环构建)
for i in tqdm(idx_iter, desc="Inferencing"):
    # A. 准备输入
    # 取 [t-512 : t] 并展平
    ctx = test_scaled[i : i+context_len].flatten().reshape(1, -1)
    
    # B. 推理
    pred_norm = model.predict(ctx).flatten()
    
    # C. 真实值
    true_norm = test_scaled[i+context_len : i+context_len+pred_len, 0]
    
    # D. 反归一化
    # 构造 Dummy 矩阵
    dummy_pred = np.zeros((pred_len, input_dims))
    dummy_pred[:, 0] = pred_norm
    pred_real = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_true = np.zeros((pred_len, input_dims))
    dummy_true[:, 0] = true_norm
    true_real = scaler.inverse_transform(dummy_true)[:, 0]
    
    all_preds.extend(pred_real)
    all_trues.extend(true_real)
    timestamps.extend(time_index[i+context_len : i+context_len+pred_len])

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

# ================= 6. 评估与保存 =================
mse = mean_squared_error(all_trues, all_preds)
mae = mean_absolute_error(all_trues, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_trues, all_preds)

print("\n" + "="*40)
print(f"📊 XGBoost 2025 Evaluation (Gap Strategy):")
print(f"   MAE  : {mae:.4f} °C")
print(f"   RMSE : {rmse:.4f} °C")
print(f"   R²   : {r2:.4f}")
print("="*40)

# 保存 NPZ
np.savez(output_npz, preds=all_preds, trues=all_trues, mse=mse, mae=mae, rmse=rmse, r2=r2)
print(f"💾 Results saved to {output_npz}")

