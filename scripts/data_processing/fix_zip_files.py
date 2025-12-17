import os
import zipfile
import glob
import shutil

# ================= 配置 =================
# 这里填你存放 raw 数据的文件夹路径
# 根据你之前的截图，应该是这个：
target_dir = "/root/autodl-tmp/project/mlpj/data/raw/era5_full_feat"

print(f"�� 正在扫描文件夹: {target_dir}")

if not os.path.exists(target_dir):
    print(f"❌ 错误: 找不到文件夹 {target_dir}")
    print("   请确认你下载的数据到底存在哪里？")
    exit()

# 扫描所有 .nc 文件 (包括那些伪装的 zip)
files = sorted(glob.glob(os.path.join(target_dir, "*.nc")))
print(f"   -> 找到 {len(files)} 个文件")

fixed_count = 0
error_count = 0

for f_path in files:
    file_name = os.path.basename(f_path)
    
    # 核心判断: 它是不是一个 ZIP 文件？
    if zipfile.is_zipfile(f_path):
        print(f"📦 发现压缩包: {file_name} -> 正在解压修复...")
        
        try:
            # 1. 创建临时解压目录
            temp_dir = f_path + "_temp_extract"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # 2. 解压
            with zipfile.ZipFile(f_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 3. 找解压出来的 .nc 文件
            # 解压出来通常叫 data.nc 或者 data_stream-oper....nc
            extracted_files = glob.glob(os.path.join(temp_dir, "*.nc"))
            
            if extracted_files:
                real_nc_path = extracted_files[0] # 拿第一个
                
                # 4. 【关键】用真的 nc 覆盖掉原来的 zip 文件
                shutil.move(real_nc_path, f_path)
                print(f"   ✅ 修复成功！")
                fixed_count += 1
            else:
                print(f"   ⚠️ 解压了但没找到 .nc 文件，跳过。")
                error_count += 1
            
            # 5. 清理临时目录
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"   ❌ 修复失败: {e}")
            error_count += 1
    else:
        # 如果不是 zip，说明它已经是正常的 nc 文件了 (或者是坏文件)
        pass

print("-" * 30)
print(f"🏁 处理完成！共修复了 {fixed_count} 个文件。")
print(f"   现在可以重新运行 batch_process_data.py 了。")
