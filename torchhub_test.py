import torch
import numpy as np
import cv2 
import csv
from efm_datasets.utils.viz import viz_depth
from efm_datasets.utils.write import write_image
import pdb

zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
intrinsics_file = 'examples/DrivingStereo/2018-10-31-06-55-01.txt'
img_file = 'examples/DrivingStereo/2018-10-31-06-55-01_2018-10-31-06-55-02-084.jpg'
# intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)
# pdb.set_trace()
# intrinsics = torch.tensor(np.load('/home/kizukuyamada/Downloads/DrivingStereo_raw/train/calib/half-image-calib/2018-10-31-06-55-01.txt')).unsqueeze(0)#ここdockerから参照できないからmntに移す
intrinsics = torch.tensor(np.load('examples/DrivingStereo/2018-10-31-06-55-01.txt')).unsqueeze(0)
# rgb = torch.tensor(cv2.imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.
# rgb = torch.tensor(cv2.imread('/home/kizukuyamada/Downloads/DrivingStereo_emaple/weather/cloudy/half/left_image/left-image-half-size/cloudy/left-image-half-size/2018-10-31-06-55-01_2018-10-31-06-55-02-084.jpg')).permute(2,0,1).unsqueeze(0)/255.
rgb = torch.tensor(cv2.imread(img_file)).permute(2,0,1).unsqueeze(0)/255.
# rgb = torch.tensor(cv2.imread('examples/frame_90000.png')).permute(2,0,1).unsqueeze(0)/255.
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

cv2.imwrite("rgb.png", rgb_np)

rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
depth_pred = zerodepth_model(rgb2, intrinsics2)
print(depth_pred)

# depth_predをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()

# #数値としてCVSに保存
# with open('pixel_values.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     for row in depth_pred_np:
#         csvwriter.writerow(row)

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

#outputを保存
cv2.imwrite("output.png", depth_pred_np)
#RGBoutputを保存
cv2.imwrite("output_RGB.png", depth_rgb_image)
#RGBoutput2を保存
write_image("output_RGB2.png", depth_rgb_image_vizdepth)
