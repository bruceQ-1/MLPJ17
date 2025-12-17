import xarray as xr
import numpy as np
import os

# 指定你要查看的文件路径
file_path = "/root/autodl-tmp/project/mlpj/era5_data_15y/shanghai_2010_01.nc"

print(f"🧐 正在检查文件: {file_path}")

if not os.path.exists(file_path):
    print("❌ 错误: 文件不存在！请检查路径或文件名。")
    exit()

try:
    # 打开 NetCDF 文件
    # engine='h5netcdf' 通常更稳定，如果报错可以去掉试试
    ds = xr.open_dataset(file_path, engine="h5netcdf")
    
    print("\n" + "="*40)
    print("📋 数据集概览 (Dataset Summary)")
    print("="*40)
    print(ds)
    
    print("\n" + "="*40)
    print("🌡️ 核心变量 't2m' (2米气温) 检查")
    print("="*40)
    
    # 检查是否有 t2m 变量
    if 't2m' in ds:
        temp_data = ds['t2m'].values
        
        # 1. 形状检查
        print(f"📏 数据形状: {temp_data.shape}")
        # 通常是 (time, latitude, longitude)
        
        # 2. 数值预览
        print(f"👀 前 5 个数值 (原始值): {temp_data.flatten()[:5]}")
        
        # 3. 统计检查 (判断单位)
        max_val = np.max(temp_data)
        min_val = np.min(temp_data)
        mean_val = np.mean(temp_data)
        
        print(f"\n📊 统计特征:")
        print(f"   Max:  {max_val:.2f}")
        print(f"   Min:  {min_val:.2f}")
        print(f"   Mean: {mean_val:.2f}")
        
        # 自动判断是 开尔文(K) 还是 摄氏度(°C)
        if mean_val > 200:
            print(f"\n💡 提示: 当前单位看起来是 [开尔文 Kelvin]。")
            print(f"   -> 对应摄氏度均值: {mean_val - 273.15:.2f} °C")
        else:
            print(f"\n💡 提示: 当前单位看起来是 [摄氏度 Celsius]。")
            
    else:
        print("⚠️ 警告: 未在文件中找到 't2m' 或 '2m_temperature' 变量。")
        print("   现有变量: ", list(ds.data_vars))

except Exception as e:
    print(f"\n❌ 读取失败: {e}")
    print("💡 可能原因: 文件可能损坏，或者其实是 ZIP 格式（如果你没运行之前的 fix_zip 脚本）。")

