import os
import re

def get_number_from_filename(filename):
    # ファイル名から数字のみを抽出する
    return int(re.sub(r'\D', '', filename))

def rename_files_in_directory(directory_path):
    # 指定されたディレクトリ内のすべてのファイルをリストアップする
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # ファイルを数字でソートする
    files.sort(key=get_number_from_filename)
    
    # すべてのファイルを再命名する
    for idx, file in enumerate(files):
        old_path = os.path.join(directory_path, file)
        # 新しいファイル名をフォーマットする（10桁のゼロ埋め）
        new_name = f'{idx:010}.png'
        new_path = os.path.join(directory_path, new_name)
        os.rename(old_path, new_path)

# ディレクトリパスを指定する
directory_path = '/data/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/image_03/data'
directory_path = '/data/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/proj_depth/map/image_03'

# 関数を呼び出してファイルを再命名する
rename_files_in_directory(directory_path)