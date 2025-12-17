'''
name: compare_all_models,py
usage: visualization
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.switch_backend('agg')

# ================= 1. 配置：定义六大模型 =================
models_config = {
    # --- 单变量基准 (Standard Protocol) ---
    "Sundial (Zero-shot)": {
        "path": "logs/eval_2025_standard_results.npz", 
        "color": "red", "style": "--"
    },
    "XGBoost (Uni-var)": {
        "path": "logs/xgboost_2025_standard_results.npz", 
        "color": "orange", "style": "--"
    },
    "LSTM (Uni-var)": {
        "path": "logs/lstm_2025_standard_results.npz", 
        "color": "blue", "style": "--"
    },
    "Bi-LSTM (Uni-var)": {
        "path": "logs/bilstm_2025_standard_results.npz", 
        "color": "purple", "style": "--"
    },
    
    # --- 多变量加强版 (Gap Strategy) ---
    "XGBoost (Rich-Feat)": {
        "path": "logs/xgboost_gap_2025_results.npz", 
        "color": "#d62728", "style": "-" # 深红实线
    },
    "LSTM (Rich-Feat)": {
        "path": "logs/lstm_gap_2025_results.npz", 
        "color": "#1f77b4", "style": "-" # 深蓝实线
    }
}

base_dir = "/root/autodl-tmp/project/mlpj/results" # 基础路径

# ================= 2. 加载数据 =================
print("正在加载 6 个模型的数据...")
results = {}
y_true = None

for name, cfg in models_config.items():
    # 尝试拼接路径，应对不同的保存习惯
    full_path = os.path.join(base_dir, cfg['path'])
    if not os.path.exists(full_path):
        # 尝试直接使用相对路径
        full_path = cfg['path']
    
    if os.path.exists(full_path):
        try:
            data = np.load(full_path)
            # 兼容不同的键名
            if 'preds' in data: pred = data['preds']
            elif 'prediction' in data: pred = data['prediction']
            elif 'lstm_pred' in data: pred = data['lstm_pred'] # 兼容 gap experiment
            elif 'xgb_pred' in data: pred = data['xgb_pred']
            else: continue
            
            # 读取一次真实值
            if y_true is None:
                if 'trues' in data: y_true = data['trues']
                elif 'truth' in data: y_true = data['truth']
            
            results[name] = pred
            print(f"   ✅ 已加载: {name}")
        except Exception as e:
            print(f"   ❌ 读取错误 {name}: {e}")
    else:
        print(f"   ⚠️ 跳过 (文件未找到): {name} -> {full_path}")

if y_true is None:
    print("❌ 致命错误：没有找到任何包含真实值(truth/trues)的文件！")
    exit()

# 生成时间轴 (2025全年)
dates = pd.date_range(start='2025-01-01', periods=len(y_true), freq='H')

# ================= 3. 计算指标表 =================
print("\n" + "="*75)
print(f"{'Model Name':<25} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
print("-" * 75)

metrics_data = []

for name, pred in results.items():
    # 对齐长度 (防止有些只有部分数据)
    min_len = min(len(pred), len(y_true))
    p = pred[:min_len]
    t = y_true[:min_len]
    
    mse = mean_squared_error(t, p)
    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(mse)
    r2 = r2_score(t, p)
    
    print(f"{name:<25} | {mae:.4f}   | {rmse:.4f}   | {r2:.4f}")
    metrics_data.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

print("="*75 + "\n")

# 保存指标到 CSV
pd.DataFrame(metrics_data).to_csv(os.path.join(base_dir, "figures/final_metrics_comparison.csv"), index=False)

# ================= 4. 绘图 (Zoom In 细节对比) =================
print("正在绘制对比图...")

# 设置绘图风格
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# --- 1月细节 (Winter) ---
ax = axes[0]
# 取前 15 天 (360小时)
limit = 360
ax.plot(dates[:limit], y_true[:limit], label="Ground Truth", color="black", linewidth=2.5, alpha=0.8)

for name, pred in results.items():
    cfg = models_config[name]
    ax.plot(dates[:limit], pred[:limit], label=name, 
            color=cfg['color'], linestyle=cfg['style'], linewidth=1.5, alpha=0.8)

ax.set_title("Model Comparison: Winter Detail (Jan 2025)", fontsize=14)
ax.set_ylabel("Temperature (°C)")
ax.legend(loc='upper right', ncol=2) # 图例分两列
ax.grid(True, alpha=0.3)

# --- 7月细节 (Summer) ---
ax = axes[1]
# 7月大概在中间
start_idx = 24 * 180 
end_idx = start_idx + 360
ax.plot(dates[start_idx:end_idx], y_true[start_idx:end_idx], label="Ground Truth", color="black", linewidth=2.5, alpha=0.8)

for name, pred in results.items():
    cfg = models_config[name]
    ax.plot(dates[start_idx:end_idx], pred[start_idx:end_idx], label=name, 
            color=cfg['color'], linestyle=cfg['style'], linewidth=1.5, alpha=0.8)

ax.set_title("Model Comparison: Summer Detail (July 2025)", fontsize=14)
ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(base_dir, "figures/final_model_battle.png")
plt.savefig(save_path, dpi=300)
print(f"✅ 对比图已保存: {save_path}")
