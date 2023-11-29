import os
import glob
import cv2

def apply_black_mask_and_save(script_directory, new_folder):
    # 新しいフォルダを作成（存在しない場合）
    new_folder_path = os.path.join(script_directory, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # スクリプトがあるディレクトリ以下のすべての.pngファイルを検索
    png_files = glob.glob(script_directory + '/**/*.jpg', recursive=True)

    first_image_shown = False

    for file_path in png_files:
        # masked_images フォルダ内のファイルはスキップ
        if new_folder in file_path:
            print(new_folder,file_path)
        else:
            # 画像を読み込む
            print(file_path)
            image = cv2.imread(file_path)
            # トリミングする領域の指定 (x, y, width, height)
            x, y, w, h = 1, 250, 1280, 200  # 例として、(100, 50)の位置から幅200、高さ200の領域をトリミング
            # 指定された範囲をトリミングする
            cropped_image = image[y:y+h, x:x+w]
            print(image.shape)

            # 最初の画像のサイズを表示し、画像を表示する
            if not first_image_shown:
                print(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")
                cv2.imshow("image", cropped_image)
                print("画像を確認して、'c'キーを押して処理を続行、または他のキーを押して終了します。")
                if cv2.waitKey(0) != ord('c'):
                    break
                first_image_shown = True

            

            # 新しいフォルダに画像を保存
            new_file_path = os.path.join(new_folder_path, os.path.basename(file_path))
            cv2.imwrite(new_file_path, cropped_image)

    cv2.destroyAllWindows()

# スクリプトがあるディレクトリを取得
# script_directory = os.path.dirname(os.path.realpath(__file__))
script_directory = "/mnt/fsx/datasets/Platooning/GTP4_vision_preview2"

# マスクを適用する範囲を指定 (左上の座標, 右下の座標)
# mask_areas = [((0, 450), (1280, 720)),((0, 0), (1280, 250))]

# 新しいフォルダ名
new_folder = 'cropped_images'

# 関数を実行
apply_black_mask_and_save(script_directory, new_folder)