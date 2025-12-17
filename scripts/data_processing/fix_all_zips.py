import os
import zipfile
import glob
import shutil

source_dir = "/root/autodl-tmp/project/mlpj/data/raw/era5_full_feat"
print(f"🔍 正在扫描并修复 ZIP 文件: {source_dir}")

files = sorted(glob.glob(os.path.join(source_dir, "*.nc")))
fixed_count = 0

for f_path in files:
    if zipfile.is_zipfile(f_path):
        try:
            temp_dir = f_path + "_temp"
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            with zipfile.ZipFile(f_path, 'r') as z:
                z.extractall(temp_dir)
            
            extracted = glob.glob(os.path.join(temp_dir, "*.nc"))
            if extracted:
                shutil.move(extracted[0], f_path)
                print(f"✅ 解压修复: {os.path.basename(f_path)}")
                fixed_count += 1
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"❌ 错误: {e}")

print(f"🏁 修复完成，共 {fixed_count} 个。")
