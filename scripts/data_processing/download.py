import cdsapi
import os

save_directory = os.path.join(os.getcwd(), "era5_data_2025")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

c = cdsapi.Client()
shanghai_area = [32.0, 120.8, 30.6, 122.3]
target_year = '2025'

for month in range(1, 13):
    month_str = f"{month:02d}"
    filename = f"shanghai_era5_land_{target_year}_{month_str}.nc"
    full_path = os.path.join(save_directory, filename)
    
    if os.path.exists(full_path):
        print(f"Skip: {filename}")
        continue

    print(f"Downloading {target_year}-{month_str}...")
    try:
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': ['2m_temperature', 'total_precipitation', '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_pressure'],
                'year': target_year,
                'month': month_str,
                'day': [str(d).zfill(2) for d in range(1, 32)],
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': shanghai_area,
                'format': 'netcdf',
            },
            full_path)
        print(f"Success: {filename}")
    except Exception as e:
        print(f"Error downloading {month_str}: {e}")
