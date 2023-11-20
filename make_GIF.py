from PIL import Image
import os
from moviepy.editor import VideoFileClip, vfx
import pdb

def convert_mp4_to_gif(mp4_file_path, gif_file_path, fps=200):
    """
    Convert an MP4 video file to a GIF file.

    :param mp4_file_path: Path to the input MP4 file.
    :param gif_file_path: Path where the output GIF should be saved.
    :param fps: Frames per second for the output GIF. Default is 10.
    """
    clip = VideoFileClip(mp4_file_path)
    clip.write_gif(gif_file_path, fps=fps)

# 使用例
# convert_mp4_to_gif("/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/30000_5_79000.mp4", "/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/output.gif")

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
#     folder_path = '/mnt/fsx/datasets/Platooning/GTP4_vision_preview2'  # 画像が保存されているフォルダのパス
#     folder_path = '/mnt/fsx/datasets/tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/inf_result/hoge'  # 画像が保存されているフォルダのパス
#     folder_path = '/mnt/fsx/datasets/DrivingStereo_tiny/2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge/inf_result/hoge'  # 画像が保存されているフォルダのパス
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
change_video_speed("/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/30000_5_79000.mp4", "/mnt/fsx/datasets/Platooning/20210916_084834000_iOSinf_result/30000_5_79000_2.mp4", speed_factor=0.2)