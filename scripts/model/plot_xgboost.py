'''
name: plot_xgboost.py
usage: plot figures from npz
       set everything before runing
input: file end up with .npy
       /root/autodl-tmp/project/mlpj/results/logs
output: figures end up with .png
       /root/autodl-tmp/project/mlpj/results
dependences: none
authors/date: Q/2025.12.16
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 无显示器模式
plt.switch_backend('agg')

# ================= 配置区域 =================
# npz文件地址
file_path = "/root/autodl-tmp/project/mlpj/results/logs/xgboost_2025_standard_results.npz"        # LSTM

# 图表名字
model_name = "XGBoost"
color_pred = 'red'  # 预测线颜色: 红色='red', 蓝色='#1f77b4', 橙色='orange'

# ================= 1. 读取 NPZ 数据 =================
print(f"正在读取文件: {file_path}")

if not os.path.exists(file_path):
    print("❌ 文件不存在！请检查路径。")
    exit()

data = np.load(file_path)

# 自动识别键名 (兼容不同脚本生成的不同 key)
keys = list(data.keys())
print(f"包含的 Keys: {keys}")

# 提取预测值
if 'preds' in data: y_pred = data['preds']
elif 'prediction' in data: y_pred = data['prediction']
else: raise ValueError("找不到预测数据 (preds/prediction)")

# 提取真实值
if 'trues' in data: y_true = data['trues']
elif 'truth' in data: y_true = data['truth']
else: raise ValueError("找不到真实数据 (trues/truth)")

# 提取指标 
mae = data['mae'] if 'mae' in data else np.nan
r2 = data['r2'] if 'r2' in data else np.nan
rmse = data['rmse'] if 'rmse' in data else np.nan

print(f"数据加载成功! 长度: {len(y_pred)}")

# ================= 2. 生成时间轴 =================
# 假设是 2025 全年数据 (8760 小时)
dates = pd.date_range(start='2025-01-01', periods=len(y_pred), freq='h')

# ================= 3. 绘图 (三段式) =================
print("正在绘图...")
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# --- 子图 1: 全年概览 ---
ax = axes[0]
ax.plot(dates, y_true, label="Ground Truth", color="green", alpha=0.5, linewidth=1)
ax.plot(dates, y_pred, label=f"{model_name} Pred", color=color_pred, alpha=0.7, linewidth=1)
title_str = f"2025 Full Year Forecast ({model_name})"
if not np.isnan(mae): title_str += f" | MAE={mae:.4f}"
if not np.isnan(r2):  title_str += f" | R2={r2:.4f}"
if not np.isnan(rmse): title_str += f" | RMSE={rmse:.4f}"
ax.set_title(title_str, fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylabel("Temp (°C)")

# --- 子图 2: 1月份细节 (前 744 小时) ---
ax = axes[1]
limit_jan = 24 * 31
ax.plot(dates[:limit_jan], y_true[:limit_jan], label="True (Jan)", color="green", linewidth=2)
ax.plot(dates[:limit_jan], y_pred[:limit_jan], label="Pred (Jan)", color=color_pred, linestyle="--", linewidth=2)
ax.set_title("Zoom in: January 2025 (Winter Detail)", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# --- 子图 3: 7月份细节 ---
ax = axes[2]
# 7月大约在第 4344 小时开始
idx_start = 24 * (31 + 28 + 31 + 30 + 31 + 30) # 6月底
idx_end = idx_start + (24 * 31)

ax.plot(dates[idx_start:idx_end], y_true[idx_start:idx_end], label="True (July)", color="green", linewidth=2)
ax.plot(dates[idx_start:idx_end], y_pred[idx_start:idx_end], label="Pred (July)", color=color_pred, linestyle="--", linewidth=2)
ax.set_title("Zoom in: July 2025 (Summer Detail)", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylabel("Temp (°C)")

# 保存
save_name = f"plot_result_{model_name.replace(' ', '_')}.png"
plt.tight_layout()
plt.savefig('/root/autodl-tmp/project/mlpj/results/figures/' + save_name, dpi=300) # dpi=300 保证高清
print(f"✅ 高清图已保存: {save_name}")
