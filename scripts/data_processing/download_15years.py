import cdsapi
import os
import time

# 保存路径
save_dir = "/root/autodl-tmp/project/mlpj/era5_data_15y"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

c = cdsapi.Client()
shanghai_area = [32.0, 120.8, 30.6, 122.3]

# 年份：2010 到 2025
years = [str(y) for y in range(2010, 2026)]

print(f"📥 开始下载任务: {years[0]} - {years[-1]} (按月分片下载)")

for year in years:
    for month in range(1, 13):
        # 格式化月份为 '01', '02' ...
        month_str = f"{month:02d}"
        filename = f"shanghai_{year}_{month_str}.nc"
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {year}-{month_str} 已存在，跳过。")
            continue

        print(f"⏳ 正在请求: {year}-{month_str} ...")
        
        try:
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': '2m_temperature',
                    'year': year,
                    'month': month_str,
                    # 每次只下载这一个月的所有天和小时
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': shanghai_area,
                    'format': 'netcdf',
                },
                filepath)
            print(f"🎉 成功: {filename}")
        except Exception as e:
            print(f"❌ 失败: {year}-{month_str} | 原因: {e}")
            # 如果是配额满，稍微等一下可能有用，但通常按月下载不会触发限制
            time.sleep(2) 
            
print("🏁 所有下载任务结束！")
