import os
import zipfile
import shutil

# 您的数据目录
data_dir = "/root/mlpj/era5_data_2025"

print(f"正在检查目录: {data_dir}")

# 遍历所有文件
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    
    # 跳过不是文件的情况
    if not os.path.isfile(file_path):
        continue

    # 判断：不管后缀名写的是什么，检查它本质是不是一个ZIP文件
    if zipfile.is_zipfile(file_path):
        print(f"发现压缩包: {filename}")
        
        try:
            # 1. 创建临时文件夹解压
            temp_dir = os.path.join(data_dir, "temp_unzip")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 2. 找到解压出来的真文件 (通常叫 data.nc)
            extracted_files = os.listdir(temp_dir)
            if extracted_files:
                real_nc_file = os.path.join(temp_dir, extracted_files[0])
                
                # 3. 这里的逻辑是：如果原文件叫 .nc 但其实是 zip，我们把解压出来的真身覆盖回去
                # 或者如果原文件是 .zip，我们把解压出来的文件改名为我们要的格式
                
                # 目标文件名 (强制改为 .nc 后缀)
                target_name = filename.replace(".zip", ".nc") 
                target_path = os.path.join(data_dir, target_name)
                
                # 移动并覆盖
                shutil.move(real_nc_file, target_path)
                print(f"  -> 成功解压并修复为: {target_name}")
                
                # 如果原文件名就是zip，解压完可以删掉原压缩包；如果是伪装的nc，已经被覆盖了
                if filename.endswith(".zip"):
                    os.remove(file_path)
            
            # 清理临时文件夹
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"  解压失败: {e}")
    else:
        print(f"文件正常 (非压缩包): {filename}")

print("修复完成！现在可以运行 process_data.py 了。")