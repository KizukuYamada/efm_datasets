import cv2
import numpy as np

def show_video_and_save_frames(video_path, start_frame, end_frame, step, save_path):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)

    # 選択されたフレーム番号を保存するためのリスト
    selected_frames = []

    # 指定されたフレーム範囲でループ
    for frame_number in range(start_frame, end_frame + 1, step):
        # フレームを読み込む
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        # フレームが正しく読み込まれたか確認
        if not ret:
            print(f"フレーム {frame_number} の取得に失敗しました。")
            continue

        # フレームを表示
        cv2.imshow(f"Frame {frame_number}", frame)
        key = cv2.waitKey(0)  # 任意のキーが押されるまで待機

        # 'c' キーが押されたら、フレーム番号をリストに追加
        if key == ord('c'):
            selected_frames.append(frame_number)

        cv2.destroyAllWindows()

    # リソースを解放
    cap.release()

    # 選択されたフレーム番号を NumPy ファイルとして保存
    np.save(save_path, np.array(selected_frames))

# 使用例
video_path = '/mnt/fsx/datasets/Platooning/20210916_084834000_iOS.mov'
start_frame = 30000
end_frame = 80000
step = 1000
save_path = 'selected_frames.npy'

show_video_and_save_frames(video_path, start_frame, end_frame, step, save_path)