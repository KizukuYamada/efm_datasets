from PIL import Image
import os
from moviepy.editor import VideoFileClip, vfx
import pdb
import cv2

def convert_mp4_to_gif_full(opencv_video_path, gif_path, fps=10):
    """
    MP4ファイルをGIFに変換する関数（全フレーム）
    :param opencv_video_path: MP4ファイルのパス
    :param gif_path: 生成されるGIFファイルのパス
    :param fps: GIFのフレームレート（デフォルトは10）
    """
    cap = cv2.VideoCapture(opencv_video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCVのBGRからRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    # フレームレートに基づいてGIFを保存
    frame_duration = 1000 / fps *5 # ミリ秒単位
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=frame_duration)

# 使用例
# opencv_video_path = '/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge/KITTI_0_0000000001_0000000005_Y0.2.mp4'  # 変換したいMP4ファイルのパス
# gif_path = '/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge/YOLOmodel/output_x.gif'            # 出力されるGIFファイルのパス
# convert_mp4_to_gif_full(opencv_video_path, gif_path)
# convert_mp4_to_gif("/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/30000_5_79000.mp4", "/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/output.gif")
# convert_mp4_to_gif("/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge/KITTI_0_0000000001_0000000005_Y0.2_cuda0.mp4"
#                    ,"/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge/output_cpu.gif")


def create_gif(folder_path, output_filename, duration=500):
    # 画像ファイルのリストを取得
    images = [img for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # ファイル名でソート
    pdb.set_trace()
    # 画像ファイルをPILイメージとして読み込む
    frames = [Image.open(os.path.join(folder_path, img)) for img in images]

    # GIFを作成して保存
    frames[0].save(output_filename, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

# if __name__ == "__main__":
#     folder_path = '/mnt/fsx/datasets/Platooning/GTP4_vision_preview2/cropped_images'  # 画像が保存されているフォルダのパス
#     # folder_path = '/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge'  # 画像が保存されているフォルダのパス
#     # folder_path = '/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge'  # 画像が保存されているフォルダのパス
#     output_filename = f'{folder_path}/output.gif'  # 出力するGIFファイルの名前
#     create_gif(folder_path, output_filename)
    
def change_video_speed(input_file_path, output_file_path, speed_factor):
    """
    Change the playback speed of a video and save it as a new file.

    :param input_file_path: Path to the input video file.
    :param output_file_path: Path where the output video file should be saved.
    :param speed_factor: Factor to change the speed of the video. 
                         Less than 1 to slow down, more than 1 to speed up.
    """
    clip = VideoFileClip(input_file_path)
    # Change the video speed
    clip = clip.fx(vfx.speedx, speed_factor)
    # Write the clip with the new speed
    clip.write_videofile(output_file_path, codec='libx264', audio_codec='aac')

# 使用例
# change_video_speed("/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge/DrivingStereo_0_0000000000_0000001118_Y0.5.mp4", 
#                    "/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge/DrivingStereo_0_0000000000_0000001118_Y0.5_2.mp4", 
#                    speed_factor=2)

import datetime
from natsort import natsorted

def create_gif(img_folder_path, output_file, timestamp_file):
    # 画像のリストを取得
    img_files = [f for f in os.listdir(img_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    img_files = natsorted(img_files)
    pdb.set_trace()
    # テキストファイルからタイムスタンプを取得
    with open(timestamp_file, 'r') as f:
        # pdb.set_trace()
        timestamps = [datetime.datetime.strptime(line.strip().split('_')[1][:-4], '%Y-%m-%d-%H-%M-%S-%f') for line in f]
    print(len(img_files), len(timestamps))
    # ソート
    # img_files = [x for _, x in sorted(zip(timestamps, img_files))]
    
    # timestamps.sort()

    # 画像を読み込む
    imgs = [Image.open(os.path.join(img_folder_path, f)) for f in img_files]

    # 各フレームの表示時間を計算
    durations = [(timestamps[i] - timestamps[i-1]).total_seconds()*1000 for i in range(1, len(timestamps))]+[1000]

    # 最初の画像を保存し、その後の画像を追加
    imgs[0].save(output_file, save_all=True, append_images=imgs[1:], duration=durations, loop=0)

# 使用例
# create_gif('/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge'
        #    , '/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge/output.gif'
        #    ,'/mnt/fsx/datasets/DrivingStereo_tiny/driving_stereo_tiny2.txt')

from moviepy.editor import VideoFileClip

def convert_gif_to_mp4(gif_path, mp4_path):
    clip = VideoFileClip(gif_path)
    clip.write_videofile(mp4_path, codec='libx264')

# 使用例
# convert_gif_to_mp4('/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge/output.gif',
                #    '/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge/output.mp4')
                
import cv2

def combine_videos(main_video_path, overlay_video_path, output_video_path, scale=0.3):
    # メインビデオとオーバーレイビデオを読み込む
    main_cap = cv2.VideoCapture(main_video_path)
    overlay_cap = cv2.VideoCapture(overlay_video_path)

    # メインビデオのプロパティを取得
    width = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = main_cap.get(cv2.CAP_PROP_FPS)

    # 出力ビデオの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret_main, frame_main = main_cap.read()
        ret_overlay, frame_overlay = overlay_cap.read()

        if not ret_main or not ret_overlay:
            break

        # オーバーレイビデオを縮小
        overlay_small = cv2.resize(frame_overlay, (0, 0), fx=scale, fy=scale)

        # オーバーレイビデオをメインビデオの左上に配置
        h, w, _ = overlay_small.shape
        frame_main[0:h, 0:w] = overlay_small

        # 結合したフレームを出力ビデオに書き込む
        out.write(frame_main)

    # リソースを解放
    main_cap.release()
    overlay_cap.release()
    out.release()

# 使用例
file = "/mnt/fsx/datasets/Platooning/20231030/20231030/nobori/20231030_065630000_iOSinf_result/x300500y250400_f3000031000_s1_after_left_c10"
main_video_path = f'{file}/output_video2.mp4'
overlay_video_path = f'{file}/depth.mp4'
output_video_path = f'{file}/combined_video.mp4'
combine_videos(main_video_path, overlay_video_path, output_video_path)