'''
name: run_lstm_gap_2025.py
usage: Train LSTM on 2010-2014, Predict 2025 
       使用七个变量训练lstm
input: /root/autodl-tmp/project/mlpj/data/processed/train_2010_2014_rich.npy
       /root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.npy
output: /root/autodl-tmp/project/mlpj/results/logs/lstm_gap_model.pth
        /root/autodl-tmp/project/mlpj/results/figures/lstm_gap_2025.png
'''
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 无显示器模式
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
    print(f"🔒 [Protocol] Random Seed set to {seed}")

set_seed(42)

# ================= 1. 配置区域 =================
# 路径 (根据你处理好的文件名)
train_npy = "/root/autodl-tmp/project/mlpj/data/processed/train_2010_2014_rich.npy"
test_npy  = "/root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.npy"
# 如果没有csv也没关系，代码里会自动生成时间轴
test_csv  = "/root/autodl-tmp/project/mlpj/data/processed/test_2025_rich.csv" 

# 保存路径
save_dir = "/root/autodl-tmp/project/mlpj/results/logs"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("/root/autodl-tmp/project/mlpj/results/figures", exist_ok=True)

# 参数
context_len = 512
pred_len = 24
input_size = 7   # 7个气象特征
hidden_size = 128
num_layers = 2
batch_size = 256
epochs = 50 
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 保存 scaler 以备后用
joblib.dump(scaler, os.path.join(save_dir, "gap_scaler.pkl"))

# ================= 3. 构建数据集 =================
class RichDataset(Dataset):
    def __init__(self, data, ctx, pred):
        self.data = torch.FloatTensor(data)
        self.ctx = ctx
        self.pred = pred
        self.len = len(data) - ctx - pred + 1
    def __len__(self): return max(0, self.len)
    def __getitem__(self, i):
        # 输入: (512, 7)
        x = self.data[i : i+self.ctx]
        # 输出: (24) -> 只要气温(第0列)
        y = self.data[i+self.ctx : i+self.ctx+self.pred, 0]
        return x, y

train_dataset = RichDataset(train_scaled, context_len, pred_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ================= 4. 定义 LSTM 模型 =================
class LSTMModel(nn.Module):
    def __init__(self, input_sz, hidden_sz, layers, out_sz):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_sz, hidden_sz, layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_sz, out_sz)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ================= 5. 训练 =================
print(f"2. Training LSTM ({epochs} Epochs)...")
model.train()
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.5f}'})
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"   Epoch {epoch+1} | Loss: {avg_loss:.5f}")

# 保存模型
torch.save(model.state_dict(), os.path.join(save_dir, "lstm_gap_model.pth"))
print("   ✅ Training Complete.")

# ================= 6. 预测 2025 =================
print("3. Predicting 2025 (Gap Test)...")
model.eval()

all_preds = []
all_trues = []
timestamps = []

# 生成 2025 时间轴 (如果没有 csv)
if os.path.exists(test_csv):
    df_test = pd.read_csv(test_csv)
    time_index = pd.to_datetime(df_test['time'])
else:
    time_index = pd.date_range(start="2025-01-01", periods=len(test_data), freq='H')

# 滚动预测 (跳过前512小时作为启动)
# 步长 24
idx_iter = range(0, len(test_scaled) - context_len - pred_len, 24)

with torch.no_grad():
    for i in tqdm(idx_iter, desc="Inferencing"):
        # 准备输入 (512, 7)
        ctx = test_scaled[i : i+context_len]
        inp = torch.FloatTensor(ctx).unsqueeze(0).to(device)
        
        # 推理
        pred_norm = model(inp).cpu().numpy().flatten()
        
        # 真实值 (气温)
        true_norm = test_scaled[i+context_len : i+context_len+pred_len, 0]
        
        # 反归一化 (Trick: 构造 dummy 矩阵)
        dummy_pred = np.zeros((pred_len, input_size))
        dummy_pred[:, 0] = pred_norm
        pred_real = scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_true = np.zeros((pred_len, input_size))
        dummy_true[:, 0] = true_norm
        true_real = scaler.inverse_transform(dummy_true)[:, 0]
        
        all_preds.extend(pred_real)
        all_trues.extend(true_real)
        
        # 记录时间 (对齐预测部分)
        timestamps.extend(time_index[i+context_len : i+context_len+pred_len])

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

# ================= 7. 评估与保存 =================
mse = mean_squared_error(all_trues, all_preds)
mae = mean_absolute_error(all_trues, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_trues, all_preds)

print("\n" + "="*40)
print(f"📊 LSTM 2025 Evaluation (Gap Strategy):")
print(f"   MAE  : {mae:.4f} °C")
print(f"   RMSE : {rmse:.4f} °C")
print(f"   R²   : {r2:.4f}")
print("="*40)

# 保存 NPZ
save_path = os.path.join(save_dir, "lstm_gap_2025_results.npz")
np.savez(save_path, preds=all_preds, trues=all_trues, mse=mse, mae=mae, rmse=rmse, r2=r2)
print(f"💾 Results saved to {save_path}")

