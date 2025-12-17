import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# 数据源路径
source_dir = "/root/autodl-tmp/project/mlpj/era5_data_15y"
output_path = "shanghai_2010_2025_full.npy"

print("1. 正在扫描文件...")
# 按文件名排序，确保时间顺序 (2010 -> ... -> 2025)
files = sorted(glob.glob(os.path.join(source_dir, "*.nc")))
print(f"   -> 找到 {len(files)} 个文件")

all_temps = []

for f in files:
    try:
        ds = xr.open_dataset(f, engine="h5netcdf")
        # 区域平均
        temp_k = ds['t2m'].mean(dim=['latitude', 'longitude']).values
        # 转摄氏度
        temp_c = temp_k - 273.15
        all_temps.append(temp_c)
        print(f"   -> 处理完成: {os.path.basename(f)} (长度: {len(temp_c)})")
    except Exception as e:
        print(f"   ❌ 读取失败 {f}: {e}")

if all_temps:
    full_data = np.concatenate(all_temps)
    np.save(output_path, full_data)
    print("-" * 30)
    print(f"✅ 巨型数据集已保存: {output_path}")
    print(f"📊 总数据量: {len(full_data)} 小时 (约 {len(full_data)/24/365:.1f} 年)")
    
    # 额外保存一份 csv 方便查看日期对应关系 (可选)
    # 假设从 2010-01-01 00:00 开始
    dates = pd.date_range(start='2010-01-01', periods=len(full_data), freq='H')
    df = pd.DataFrame({'time': dates, 'temperature': full_data})
    df.to_csv("shanghai_2010_2025.csv", index=False)
    print("   -> 同时保存了 CSV 索引表")
else:
    print("❌ 没有处理任何数据")
