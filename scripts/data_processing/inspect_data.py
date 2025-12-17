import numpy as np
import pandas as pd
import os

# 定义你的文件路径
npy_path = "shanghai_2010_2025_full.npy"
csv_path = "shanghai_2010_2025.csv"

print("="*40)
print("🔍 数据集体检报告")
print("="*40)

# 1. 检查 .npy 文件 (纯数值)
if os.path.exists(npy_path):
    print(f"\n[1] 正在检查 NPY 文件: {npy_path}")
    try:
        data = np.load(npy_path)
        print(f"   ✅ 加载成功")
        print(f"   📏 数据形状 (Shape): {data.shape}")
        print(f"      -> 意味着有 {data.shape[0]} 个小时的数据")
        print(f"      -> 约等于 {data.shape[0] / 24 / 365:.2f} 年")
        print(f"   🔢 数据类型 (Dtype): {data.dtype}")
        
        print(f"\n   👀 数据预览:")
        print(f"      前 5 个数据: {data[:5]}")
        print(f"      后 5 个数据: {data[-5:]}")
        
        print(f"\n   📊 统计特征 (检查是否有异常值):")
        print(f"      最高温 (Max):  {np.max(data):.2f} °C")
        print(f"      最低温 (Min):  {np.min(data):.2f} °C")
        print(f"      平均温 (Mean): {np.mean(data):.2f} °C")
        
        # 简单检查是否有离谱数据 (比如 100度 或 -100度)
        if np.max(data) > 60 or np.min(data) < -30:
            print("      ⚠️ 警告: 温度数据似乎超出正常范围，请检查是否单位错误(如开尔文)！")
        else:
            print("      ✅ 温度范围看起来是正常的摄氏度。")
            
    except Exception as e:
        print(f"   ❌ 读取出错: {e}")
else:
    print(f"   ❌ 找不到文件: {npy_path}")

# 2. 检查 .csv 文件 (带时间标签)
if os.path.exists(csv_path):
    print(f"\n[2] 正在检查 CSV 文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ 加载成功")
        print(f"   📏 行数: {len(df)}")
        print(f"   📅 时间范围:")
        print(f"      开始: {df['time'].iloc[0]}")
        print(f"      结束: {df['time'].iloc[-1]}")
        
        print(f"\n   👀 表格预览 (前3行):")
        print(df.head(3))
    except Exception as e:
        print(f"   ❌ 读取出错: {e}")
else:
    print(f"\n[2] 找不到 CSV 文件: {csv_path} (不影响训练，主要用于核对时间)")

print("\n" + "="*40)
