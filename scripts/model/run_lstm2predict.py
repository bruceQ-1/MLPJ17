'''
name: run_lstm2predict.py
usage: use LSTM to train and predict
input: /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy
       /root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv
output: /root/autodl-tmp/project/mlpj/results/logs/lstm_2025_standard_results.npz
authors/date: Q/2025.12.16
'''

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 设置无显示器模式
plt.switch_backend('agg')

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

set_seed(42)

# ================= 1. 配置区域 =================
# 路径 (与 Sundial 保持完全一致)
data_npy = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_full.npy"
data_csv = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025.csv"

# 输出路径
save_dir = "/root/autodl-tmp/project/mlpj/results/logs"
os.makedirs(save_dir, exist_ok=True)
output_npz = os.path.join(save_dir, "lstm_2025_standard_results.npz")
output_img = "lstm_2025_standard.png"

# 标准化参数
context_len = 512
pred_len = 24
stride = 24

# 训练参数
batch_size = 256
epochs = 50 
learning_rate = 0.001
hidden_size = 128
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 规则: 只能用训练集 fit
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data_raw)

# 转换全量数据
data_scaled = scaler.transform(data_values) # (N, 1)
print(f"   -> [Protocol] Scaler fitted on Train set (2010-2024).")

# ================= 3. 构建数据集 =================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_len, pred_len):
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.pred_len = pred_len
        self.len = len(data) - context_len - pred_len + 1
        
    def __len__(self):
        return max(0, self.len)
    
    def __getitem__(self, idx):
        # 输入: (512, 1)
        x = self.data[idx : idx + self.context_len]
        # 输出: (24) -> 展平为 (24,)
        y = self.data[idx + self.context_len : idx + self.context_len + self.pred_len].squeeze()
        return x, y

# 只用训练集部分构建 DataLoader
train_dataset = TimeSeriesDataset(data_scaled[:test_start_idx], context_len, pred_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
print(f"   -> Training Samples: {len(train_dataset)}")

# ================= 4. 定义 LSTM 模型 =================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq, feature)
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] 
        predictions = self.fc(last_step_out)
        return predictions

model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ================= 5. 训练模型 =================
print(f"2. Training LSTM ({epochs} Epochs)...")
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    # 简单的进度显示
    if (epoch+1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs} running...")
        
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

print("   ✅ Training Complete.")

# ================= 6. 滚动预测 (Standard Rolling) =================
print(f"3. [Protocol] Rolling Forecast (Stride={stride})...")
model.eval()

all_preds = []
all_trues = []
timestamps = []

# 迭代器: 从 2025-01-01 开始
idx_iter = range(test_start_idx, len(data_values) - pred_len, stride)

with torch.no_grad():
    for current_idx in tqdm(idx_iter, desc="Inferencing"):
        # A. 准备输入 (归一化后的数据)
        # 取 [t-512 : t]
        history_norm = data_scaled[current_idx - context_len : current_idx]
        input_tensor = torch.FloatTensor(history_norm).unsqueeze(0).to(device) # (1, 512, 1)
        
        # B. 推理
        pred_norm = model(input_tensor).cpu().numpy().flatten() # (24,)
        
        # C. 真实值
        true_norm = data_scaled[current_idx : current_idx + pred_len].flatten()
        
        # D. 反归一化 (使用全局 Scaler)
        pred_real = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        true_real = scaler.inverse_transform(true_norm.reshape(-1, 1)).flatten()
        
        all_preds.extend(pred_real)
        all_trues.extend(true_real)
        timestamps.extend(df['time'].iloc[current_idx : current_idx + pred_len])

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

# ================= 7. 计算指标 =================
mse = mean_squared_error(all_trues, all_preds)
mae = mean_absolute_error(all_trues, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_trues, all_preds)

print("\n" + "="*40)
print(f"📊 LSTM 2025 Evaluation Report:")
print(f"   MAE  : {mae:.4f} °C")
print(f"   RMSE : {rmse:.4f} °C")
print(f"   MSE  : {mse:.4f}")
print(f"   R²   : {r2:.4f}")
print("="*40)

# ================= 8. 保存结果 =================
np.savez(output_npz, preds=all_preds, trues=all_trues, mse=mse, mae=mae, rmse=rmse, r2=r2)
print(f"💾 Results saved to {output_npz}")

# 简单绘图
plt.figure(figsize=(12, 6))
plt.plot(timestamps, all_trues, label="Ground Truth", color="green", alpha=0.5)
plt.plot(timestamps, all_preds, label="LSTM Prediction", color="blue", alpha=0.6, linewidth=1)
plt.title(f"LSTM Standard Protocol: MAE={mae:.2f}, R2={r2:.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_img)
print(f"📈 Plot saved to {output_img}")
