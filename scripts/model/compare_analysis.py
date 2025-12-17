'''
name: compare_analysis_v2.py
usage: compare 5 models including Bi-LSTM
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score

plt.switch_backend('agg')

# ================= 配置 =================

models_config = {
    "Sundial":        "logs/eval_2025_standard_results.npz",
    "XGBoost (Uni)":  "logs/xgboost_2025_standard_results.npz",
    "Bi-LSTM (Uni)":  "logs/bilstm_2025_standard_results.npz", 
    "XGBoost (Rich)": "logs/xgboost_gap_2025_results.npz",
    "LSTM (Rich)":    "logs/lstm_gap_2025_results.npz",
}

base_dir = "/root/autodl-tmp/project/mlpj/results"
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# ================= 加载数据 =================
print("Loading data...")
results = {}
y_true = None

for name, path in models_config.items():
    full_path = os.path.join(base_dir, path)
    if not os.path.exists(full_path):
        # 尝试相对路径
        if os.path.exists(path): full_path = path
        else: 
            print(f"⚠️ 跳过: 找不到文件 {path}")
            continue
            
    try:
        data = np.load(full_path)
        # 兼容各种 key 名
        if 'preds' in data: p = data['preds']
        elif 'prediction' in data: p = data['prediction']
        elif 'xgb_pred' in data: p = data['xgb_pred']
        elif 'lstm_pred' in data: p = data['lstm_pred']
        else: continue
        
        results[name] = p
        print(f"✅ Loaded: {name}")
        
        if y_true is None:
            if 'trues' in data: y_true = data['trues']
            elif 'truth' in data: y_true = data['truth']
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        pass

if y_true is None: 
    print("No truth data found!")
    exit()

# 生成时间索引 (用于分时分析)
dates = pd.date_range(start='2025-01-01', periods=len(y_true), freq='H')

# ================= 1. 误差分布图 (Residual Histogram) =================
print("1. Plotting Error Distribution...")
plt.figure(figsize=(10, 6))

for name, pred in results.items():
    # 对齐长度
    min_len = min(len(pred), len(y_true))
    # 计算残差: 预测 - 真实
    residuals = pred[:min_len] - y_true[:min_len]
    # 画密度曲线 (KDE)
    sns.kdeplot(residuals, label=f"{name}", fill=True, alpha=0.1, linewidth=2)

plt.axvline(0, color='black', linestyle='--', linewidth=1) # 0线
plt.title("Error Distribution (Bias Analysis)", fontsize=14)
plt.xlabel("Error (°C)  [<0: Underestimate, >0: Overestimate]")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "analysis_error_dist.png"))

# ================= 2. 散点回归图 (Scatter Regression) =================
print("2. Plotting Scatter Regression...")
# 动态调整画布宽度
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
if len(results) == 1: axes = [axes]

for ax, (name, pred) in zip(axes, results.items()):
    min_len = min(len(pred), len(y_true))
    y_t = y_true[:min_len]
    y_p = pred[:min_len]

    # 降采样防卡顿
    if len(y_t) > 2000:
        sample_idx = np.random.choice(len(y_t), 2000, replace=False)
        y = y_t[sample_idx]
        p = y_p[sample_idx]
    else:
        y, p = y_t, y_p
    
    # 散点
    ax.scatter(y, p, alpha=0.2, s=10, color='#1f77b4')
    
    # 对角线
    mi = min(y.min(), p.min())
    ma = max(y.max(), p.max())
    ax.plot([mi, ma], [mi, ma], 'r--', linewidth=2, label='Perfect Fit')
    
    # 计算 R2
    r2 = r2_score(y, p)
    ax.set_title(f"{name}\n$R^2$ = {r2:.4f}")
    ax.set_xlabel("Ground Truth (°C)")
    ax.set_ylabel("Predicted (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "analysis_scatter.png"))

# ================= 3. 分时误差图 (Error by Hour) =================
print("3. Plotting Error by Hour...")
plt.figure(figsize=(12, 6))

hours = np.arange(24)

for name, pred in results.items():
    min_len = min(len(pred), len(y_true))
    # 计算绝对误差
    abs_error = np.abs(pred[:min_len] - y_true[:min_len])
    
    temp_df = pd.DataFrame({'Error': abs_error, 'Hour': dates[:min_len].hour})
    hourly_mae = temp_df.groupby('Hour')['Error'].mean()
    
    plt.plot(hourly_mae.index, hourly_mae.values, marker='o', label=name, linewidth=2)

plt.title("Model Performance by Hour of Day (Diurnal Cycle)", fontsize=14)
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xticks(hours)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "analysis_hourly_error.png"))

print("✅ Analysis Complete. Check 'figures' folder.")