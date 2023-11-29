import os

def create_file_list(base_path, output_file, prefix_to_remove):
    # ディレクトリパスを定義
    dir1 = os.path.join(base_path, "image_02/data_raw")
    dir2 = os.path.join(base_path, "image_03/data_raw")

    # 出力ファイルを開く
    with open(output_file, 'w') as f:
        # 両方のディレクトリのファイルリストを取得
        files1 = sorted(os.listdir(dir1))
        files2 = sorted(os.listdir(dir2))

        # ファイル名を一行ずつ書き込む
        for file1, file2 in zip(files1, files2):
            if file1.endswith('.jpg') and file2.endswith('.jpg'):
                # full_path1 = os.path.join(base_path, 'image_02/data', file1)
                # full_path2 = os.path.join(base_path, 'image_03/data', file2)
                full_path1 = file1
                # プレフィックスを取り除く
                relative_path1 = full_path1.replace(prefix_to_remove, '')
                # relative_path2 = full_path2.replace(prefix_to_remove, '')
                # line = f"{relative_path1} {relative_path2}\n"
                line = f"{relative_path1}\n"
                f.write(line)

# ベースパス、出力ファイル名、取り除くプレフィックスを指定
base_path = "/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge"
prefix_to_remove = "/mnt/fsx/datasets/DrivingStereo_tiny/"
output_file = f"{prefix_to_remove}driving_stereo_tiny2.txt"

create_file_list(base_path, output_file, prefix_to_remove)