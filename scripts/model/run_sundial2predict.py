'''
name: run_sundial2predict.py
usage: predict 2025 tep by Sundial
imput: /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy
       /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv
output: /root/autodl-tmp/project/mlpj/results/logs/eval_2025_standard_results.npz
authors/date: Q/2025.12.14
'''
import os
import random
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
# 引入标准化所需的库
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 0. 标准化协议：随机种子 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Protocol] Random Seed set to {seed}")

set_seed(42)  # 锁定随机性

# ================= 1. 配置区域 =================
model_path = "/root/autodl-tmp/models/sundial-base-128m"
data_npy = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy"
data_csv = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv"

# 标准化参数
context_len = 512
pred_len = 24
stride = 24

# ================= 2. 数据加载与标准化预处理 =================
print("1. [Protocol] Loading & Preprocessing Data...")
if not os.path.exists(data_npy) or not os.path.exists(data_csv):
    print("❌ Data not found.")
    exit()

# 加载原始数据
data_values = np.load(data_npy) # (N,)
if data_values.ndim == 1:
    data_values = data_values.reshape(-1, 1) # 确保是 (N, 1)

df = pd.read_csv(data_csv)
df['time'] = pd.to_datetime(df['time'])

# --- 严格划分训练集与测试集 ---
split_date = pd.Timestamp("2025-01-01 00:00:00")
train_mask = df['time'] < split_date

# 训练集: 2010-2024
train_data_raw = data_values[train_mask]

# 测试起点索引
test_start_idx = np.sum(train_mask)
print(f"   -> Train Data End: {df['time'].iloc[test_start_idx-1]}")
print(f"   -> Test Start:     {df['time'].iloc[test_start_idx]}")

# --- 统一预处理: MinMaxScaler (0~1) ---
# 规则: 只能用训练集(2010-2024) fit，防止未来数据泄露
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data_raw)

# 转换全量数据
data_scaled = scaler.transform(data_values).flatten() # (N,)
print(f"   -> [Protocol] Scaler fitted on Train set (2010-2024).")
print(f"   -> Scaled Data Range: {data_scaled.min():.3f} ~ {data_scaled.max():.3f}")

# ================= 3. 加载模型 =================
print("2. Loading Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(device)
    model.eval()
    print(f"   ✅ Model loaded on {device}")
except Exception as e:
    print(f"❌ Load failed: {e}")
    exit()

# ================= 4. 滚动预测 (Standard Rolling) =================
print(f"3. [Protocol] Rolling Forecast (Stride={stride})...")

all_preds = []
all_trues = []
timestamps = []

# 迭代器: 从 2025-01-01 开始
idx_iter = range(test_start_idx, len(data_values) - pred_len, stride)

for current_idx in tqdm(idx_iter, desc="Inferencing"):
    # A. 准备输入 (从全局归一化后的数据中取)
    # 规则: 直接取 data_scaled，不再做局部归一化
    history_norm = data_scaled[current_idx - context_len : current_idx]
    
    input_tensor = torch.tensor(history_norm, dtype=torch.float32).unsqueeze(0).to(device)
    
    # B. 模型推理
    with torch.no_grad():
        # 固定采样行为 (虽然设置了seed，但显式控制更安全)
        # Sundial 是生成模型，这里为了稳定取 3 次中位数
        batch_preds = []
        for _ in range(3):
            output = model.generate(input_tensor, max_new_tokens=pred_len, num_samples=1)
            pred_slice = output[:, -pred_len:]
            batch_preds.append(pred_slice.cpu().numpy())
        
        # 得到归一化后的预测值 (0~1之间)
        pred_norm = np.median(np.array(batch_preds), axis=0).flatten()
    
    # C. 获取真实值 (归一化后的)
    true_norm = data_scaled[current_idx : current_idx + pred_len]
    
    # D. 反归一化 (Inverse Transform)
    # 规则: 使用全局 Scaler 还原
    pred_real = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    true_real = scaler.inverse_transform(true_norm.reshape(-1, 1)).flatten()
    
    all_preds.extend(pred_real)
    all_trues.extend(true_real)
    timestamps.extend(df['time'].iloc[current_idx : current_idx + pred_len])

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

# ================= 5. 计算指标 (Standard Metrics) =================
mse = mean_squared_error(all_trues, all_preds)
mae = mean_absolute_error(all_trues, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_trues, all_preds)

print("\n" + "="*40)
print(f"📊 2025 Evaluation Report (Standard Protocol):")
print(f"   MAE  : {mae:.4f} °C")
print(f"   RMSE : {rmse:.4f} °C")
print(f"   MSE  : {mse:.4f}")
print(f"   R²   : {r2:.4f}")
print("="*40)

# ================= 6. 保存结果 =================
save_path = "/root/autodl-tmp/project/mlpj/results/logs/eval_2025_standard_results.npz"
np.savez(save_path, preds=all_preds, trues=all_trues, mse=mse, mae=mae, rmse=rmse, r2=r2)
print(f"💾 Results saved to {save_path}")
