'''
name: batch_process_data.py
usage: 处理数据
'''
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
source_dir = "/root/autodl-tmp/project/mlpj/data/raw/era5_full_feat" 

output_dir = "/root/autodl-tmp/project/mlpj/data/processed"
os.makedirs(output_dir, exist_ok=True)

def process_years(years, save_name):
    print(f"\n🚀 正在处理年份: {years} -> {save_name}")
    files = []
    for y in years:
        # 搜索该年份的所有 nc 文件
        found = glob.glob(os.path.join(source_dir, f"*{y}*.nc"))
        files.extend(found)
    
    files = sorted(files)
    if not files:
        print(f"   ❌ 未找到年份 {years} 的文件，跳过！")
        return

    print(f"   -> 找到 {len(files)} 个文件，开始提取特征...")
    
    all_data = []
    timestamps = []
    
    for f in tqdm(files):
        try:
            ds = xr.open_dataset(f, engine="h5netcdf")
            
            # --- 提取 7 大特征 ---
            # 1. 气温 (K -> C)
            t2m = ds['t2m'].mean(dim=['latitude', 'longitude']).values - 273.15
            # 2. 露点 (K -> C)
            d2m = ds['d2m'].mean(dim=['latitude', 'longitude']).values - 273.15
            # 3. 降雨 (m -> mm)
            tp  = ds['tp'].mean(dim=['latitude', 'longitude']).values * 1000
            # 4. 气压 (Pa -> hPa)
            sp  = ds['sp'].mean(dim=['latitude', 'longitude']).values / 100
            # 5. 辐射 (J/m2)
            ssrd= ds['ssrd'].mean(dim=['latitude', 'longitude']).values
            # 6. 土壤水 (0-1)
            swvl1=ds['swvl1'].mean(dim=['latitude', 'longitude']).values
            # 7. 风速 (合成 m/s)
            u10 = ds['u10'].mean(dim=['latitude', 'longitude']).values
            v10 = ds['v10'].mean(dim=['latitude', 'longitude']).values
            wind = np.sqrt(u10**2 + v10**2)
            
            # 堆叠 (Time, 7)
            batch = np.stack([t2m, d2m, tp, sp, ssrd, swvl1, wind], axis=1)
            all_data.append(batch)
            
            # 生成时间轴 (辅助)
            # 简单生成: 假设每个文件是一个月
            # 这里不直接读 ds.time 因为格式可能乱，我们后面统一生成
            
            ds.close()
        except Exception as e:
            print(f"   ⚠️ 读取失败 {os.path.basename(f)}: {e}")

    if all_data:
        full_arr = np.concatenate(all_data, axis=0)
        save_path = os.path.join(output_dir, save_name)
        np.save(save_path, full_arr)
        print(f"   ✅ 保存成功: {save_path} | Shape: {full_arr.shape}")
        
        # 保存一份 CSV 方便查时间
        # 假设数据是连续的，我们根据长度反推时间
        start_year = years[0]
        dates = pd.date_range(start=f'{start_year}-01-01', periods=len(full_arr), freq='H')
        df = pd.DataFrame(full_arr, columns=['t2m', 'd2m', 'tp', 'sp', 'ssrd', 'swvl1', 'wind'])
        df.insert(0, 'time', dates)
        csv_path = save_path.replace('.npy', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"   ✅ CSV 已保存: {csv_path}")

# ================= 执行处理 =================
# 1. 训练集: 2010, 2011, 2012, 2013, 2014
train_years = [str(y) for y in range(2010, 2015)]
process_years(train_years, "train_2010_2014_rich.npy")

# 2. 测试集: 2025
test_years = ["2025"]
process_years(test_years, "test_2025_rich.npy")
