import os

def get_first_and_last_file_sorted(directory):
    # ディレクトリ内の全ファイルのパスを取得
    all_files = [os.path.join(directory, file) for file in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, file))]

    # ファイル名でソート
    all_files.sort()

    # 最初と最後のファイルを返す
    return all_files[0] if all_files else None, all_files[-1] if all_files else None


# 指定されたパス
path = '/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/image_02/data/'

# 最初と最後のファイル名を取得
first_file, last_file = get_first_and_last_file_sorted(path)
print("First file:", first_file)
print("Last file:", last_file)