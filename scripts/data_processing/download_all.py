

import cdsapi
import os
import time

# 存放在名为 full_feat 的新文件夹
save_dir = "/root/autodl-tmp/project/mlpj/data/raw/era5_full_feat"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

c = cdsapi.Client()
shanghai_area = [32.0, 120.8, 30.6, 122.3]
years = [str(y) for y in range(2025, 2026)]

variables = [
    '2m_temperature',
    'total_precipitation',
    'surface_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    'surface_solar_radiation_downwards',
    'volumetric_soil_water_layer_1',
]

print(f"📥 开始下载全特征数据集 (2025-2025)...")

for year in years:
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filename = f"shanghai_all_{year}_{month_str}.nc"
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {filename} 已存在，跳过。")
            continue

        print(f"⏳ 请求中: {year}-{month_str} ...")
        try:
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': variables,
                    'year': year,
                    'month': month_str,
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': shanghai_area,
                    'format': 'netcdf',
                },
                filepath)
            print(f"🎉 下载成功: {filename}")
        except Exception as e:
            print(f"❌ 下载失败: {year}-{month_str} | {e}")
            time.sleep(2)

print("🏁 下载完成。")
