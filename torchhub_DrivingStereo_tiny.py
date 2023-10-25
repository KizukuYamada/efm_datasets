import torch
import numpy as np
import cv2 
import csv
from efm_datasets.utils.viz import viz_depth
from efm_datasets.utils.write import write_image
import os
import pdb

zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
# # sunny
# name = 'DrivinfStereo/sunny'
# intrinsics_file = '/data/datasets/DrivingStereo_emaple/calib/half-image-calib/2018-10-19-09-30-39.txt'
# img_file = '/data/datasets/DrivingStereo_emaple/weather/sunny/half/left_image/left-image-half-size/sunny/left-image-half-size/2018-10-19-09-30-39_2018-10-19-09-31-36-438.jpg'
# cloudy
# name = 'DrivinfStereo/cloudy'
# intrinsics_file = '/data/datasets/DrivingStereo_emaple/calib/half-image-calib/2018-10-31-06-55-01.txt'
# img_file = '/data/datasets/DrivingStereo_emaple/weather/cloudy/half/left_image/left-image-half-size/cloudy/left-image-half-size/2018-10-31-06-55-01_2018-10-31-06-56-22-731.jpg'
# # rainy
# name = 'DrivinfStereo/rainy'
# intrinsics_file = '/data/datasets/DrivingStereo_emaple/calib/half-image-calib/2018-08-17-09-45-58.txt'
# img_file = '/data/datasets/DrivingStereo_emaple/weather/rainy/half/left_image/left-image-half-size/rainy/left-image-half-size/2018-08-17-09-45-58_2018-08-17-10-11-18-951.jpg'
# # foggy
# name = 'DrivinfStereo/foggy'
# intrinsics_file = '/data/datasets/DrivingStereo_emaple/calib/half-image-calib/2018-10-25-07-37-26.txt'
# img_file = '/data/datasets/DrivingStereo_emaple/weather/foggy/half/left_image/left-image-half-size/foggy/left-image-half-size/2018-10-25-07-37-26_2018-10-25-07-45-25-505.jpg'
def read_p_rect_101(file_path):
    with open(file_path, 'r') as file:
        # ファイルの各行を読み取る
        lines = file.readlines()
        
    for line in lines:
        # 行がP_rect_101を含むかどうかを確認する
        if 'P_rect_101:' in line:
            # P_rect_101の行を返す
            return line.strip()  # strip()は、行の前後の空白を削除する


# 関数を呼び出してP_rect_101の行を取得する
p_rect_101_line = read_p_rect_101(intrinsics_file)
print(p_rect_101_line)
p_rect_101_values = [float(value) for value in p_rect_101_line.split()[1:]]  # 最初の要素をスキップ]
print(p_rect_101_values)
p_rect_101_matrix = np.array(p_rect_101_values).reshape(3, 4)
print(p_rect_101_matrix)
intrinsics = torch.tensor(p_rect_101_matrix, dtype=torch.float32).unsqueeze(0)
rgb = torch.tensor(cv2.imread(img_file), dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255.
print(rgb.shape)

# original画像をNumPy配列に変換して表示
rgb_np = cv2.imread(img_file)
# rgb.squeeze().cpu().detach().numpy()
def resize_rgb_intrinsics(rgb, intrinsics):
    # rgbの幅と高さを取得
    rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
    print(rgb_h,rgb_w)
    # rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
    resized_tensor = torch.nn.functional.interpolate(rgb, size=(384, 640), mode='bilinear', align_corners=False)
    print(rgb.shape)
    intrinsics2 = intrinsics.clone()
    fx_new = intrinsics[0,0,0].item()*640/rgb_w
    cx_new = intrinsics[0,0,2].item()*640/rgb_w
    fy_new = intrinsics[0,1,1].item()*384/rgb_h
    cy_new = intrinsics[0,1,2].item()*384/rgb_h
    intrinsics2[0,0,0] = fx_new
    intrinsics2[0,0,2] = cx_new
    intrinsics2[0,1,1] = fy_new
    intrinsics2[0,1,2] = cy_new
    print("intrinsics2:", intrinsics2)
    print("intrinsics:", intrinsics)
    return resized_tensor, intrinsics2



rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
depth_pred = zerodepth_model(rgb2, intrinsics2)
print(depth_pred)

# depth_predをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()

# depth_pred_npの最大値と最小値を取得
depth_pred_max = np.max(depth_pred_np)
depth_pred_min = np.min(depth_pred_np)

# depthをRGB画像に変換する関数
def depth_to_rgb(depth_data, depth_min, depth_max):
    # depthを最大値と最小値でクリップ（範囲内に収める）
    clipped_depth = np.clip(depth_data, depth_min, depth_max)
    # 深度を0から1の範囲にスケーリング
    normalized_depth = (clipped_depth - depth_min) / (depth_max - depth_min)
    # 0から255の範囲にスケーリング
    depth_scaled = (normalized_depth * 255).astype(np.uint8)
    # カラーマップを適用してRGB画像に変換
    depth_rgb = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    return depth_rgb

# viz_depthを使ってdepthを可視化
# pdb.set_trace()
depth_rgb_image_vizdepth = viz_depth(depth_pred) 
print(depth_rgb_image_vizdepth.shape)   

# depthをRGBに変換
depth_rgb_image = depth_to_rgb(depth_pred_np, depth_pred_min, depth_pred_max)

## フォルダを作成
if not os.path.exists(f"Infe_data/{name}"):
    os.mkdir(f"Infe_data/{name}")
os.chmod(f"Infe_data/{name}", 0o777)

#オリジナルイメージを保存
cv2.imwrite(f"Infe_data/{name}/rgb_input.png", rgb_np)
#outputを保存
# cv2.imwrite(f"Infe_data/{name}/depth_color.png",  depth_pred_np)
#RGBoutputを保存
cv2.imwrite(f"Infe_data/{name}/depth_color.png", depth_rgb_image)
#RGBoutput2を保存
write_image(f"Infe_data/{name}/depth_c_inv.png", depth_rgb_image_vizdepth)

