import os
import zipfile
import glob
import shutil

# 数据存放路径
source_dir = "/root/autodl-tmp/project/mlpj/era5_data_15y"

print(f"🔍 正在扫描 {source_dir} 下的文件...")
files = sorted(glob.glob(os.path.join(source_dir, "*.nc")))
print(f"   -> 找到 {len(files)} 个文件")

fixed_count = 0
error_count = 0

for f_path in files:
    f_name = os.path.basename(f_path)
    
    # 1. 检查是不是 ZIP 文件
    if zipfile.is_zipfile(f_path):
        print(f"📦 发现伪装文件: {f_name} (实为 ZIP)")
        
        try:
            # 创建临时解压目录
            temp_dir = f_path + "_temp_extract"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # 解压
            with zipfile.ZipFile(f_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 找到解压出来的真正的 .nc 文件 (通常叫 data.nc 或 data_stream-oper...nc)
            extracted_ncs = glob.glob(os.path.join(temp_dir, "*.nc"))
            
            if extracted_ncs:
                real_nc_path = extracted_ncs[0] # 取第一个
                
                # 关键步骤：用真的 nc 文件覆盖掉原来的 zip 文件
                shutil.move(real_nc_path, f_path)
                print(f"   ✅ 修复成功！已替换为真实 NetCDF 文件。")
                fixed_count += 1
            else:
                print(f"   ⚠️ 解压后没找到 .nc 文件，跳过。")
                error_count += 1
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"   ❌ 修复失败: {e}")
            error_count += 1
    else:
        # 如果不是 ZIP，可能是真正的 nc，或者是纯文本报错文件
        # 我们简单读一下开头，看看是不是 'CDF' 或 'HDF' 开头
        with open(f_path, 'rb') as f:
            header = f.read(4)
        
        if header.startswith(b'CDF') or header[1:4] == b'HDF':
            # print(f"   🆗 {f_name} 看起来是正常的。")
            pass
        else:
            print(f"   ⚠️ {f_name} 既不是 ZIP 也不是标准 NC (Header: {header})")
            error_count += 1

print("-" * 30)
print(f"🏁 处理完成。修复了 {fixed_count} 个文件，异常 {error_count} 个。")
print("请重新运行 process_15years.py")
