import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

source_dir = "/root/autodl-tmp/project/mlpj/data/raw/era5_full_feat"
output_npy = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_all_features.npy"
output_csv = "/root/autodl-tmp/project/mlpj/data/processed/shanghai_2010_2025_all_features.csv"

print("1. 扫描文件...")
files = sorted(glob.glob(os.path.join(source_dir, "*.nc")))

all_data = []

for f in files:
    try:
        ds = xr.open_dataset(f, engine="h5netcdf")
        
        # --- 1. 基础变量提取 (区域平均) ---
        t2m = ds['t2m'].mean(dim=['latitude', 'longitude']).values - 273.15  # 气温 (C)
        d2m = ds['d2m'].mean(dim=['latitude', 'longitude']).values - 273.15  # 露点 (C)
        tp  = ds['tp'].mean(dim=['latitude', 'longitude']).values * 1000     # 降雨 (mm)
        sp  = ds['sp'].mean(dim=['latitude', 'longitude']).values / 100      # 气压 (hPa)
        ssrd= ds['ssrd'].mean(dim=['latitude', 'longitude']).values          # 辐射 (J/m2)
        swvl1=ds['swvl1'].mean(dim=['latitude', 'longitude']).values         # 土壤水 (m3/m3)
        
        # --- 2. 风速合成 ---
        u10 = ds['u10'].mean(dim=['latitude', 'longitude']).values
        v10 = ds['v10'].mean(dim=['latitude', 'longitude']).values
        wind_speed = np.sqrt(u10**2 + v10**2)  # 合成风速 (m/s)
        
        # --- 3. 堆叠特征 ---
        # 顺序: [气温, 露点, 降雨, 气压, 辐射, 土壤水, 风速] (共7维)
        batch = np.stack([t2m, d2m, tp, sp, ssrd, swvl1, wind_speed], axis=1)
        all_data.append(batch)
        
        print(f"   ✅ 处理: {os.path.basename(f)}")
        
    except Exception as e:
        print(f"   ❌ 读取失败: {os.path.basename(f)} | {e}")

if all_data:
    full_data = np.concatenate(all_data, axis=0)
    
    # 保存 NPY
    np.save(output_npy, full_data)
    print(f"\n💾 NPY 已保存: {output_npy} | Shape: {full_data.shape}")
    
    # 保存 CSV (带表头)
    dates = pd.date_range(start='2010-01-01', periods=len(full_data), freq='H')
    cols = ['temperature', 'dewpoint', 'precip', 'pressure', 'radiation', 'soil_water', 'wind_speed']
    df = pd.DataFrame(full_data, columns=cols)
    df.insert(0, 'time', dates)
    df.to_csv(output_csv, index=False)
    print(f"💾 CSV 已保存: {output_csv}")
    
    # 【关键】提示删除原始文件以节省空间
    print("\n⚠️ 提示: 您的硬盘只有 35GB，建议现在删除原始 .nc 文件。")
    print(f"   运行: rm -rf {source_dir}")

else:
    print("❌ 无数据处理")
