import torch
import numpy as np
import cv2 
import csv
import pdb
from efm_datasets.utils.config import read_config
from efm_datasets.utils.setup import setup_dataset
from efm_datasets.utils.data import make_batch, fold_batch, get_from_dict, interleave_dict, modrem
from efm_datasets.utils.viz import viz_depth
from efm_datasets.utils.write import write_image
from scripts.display.DisplayDataset import DisplayDataset
import sys
import os

config = sys.argv[1]
print(config)
name = sys.argv[2]
cfg = read_config(config)

## 設定を上書き
time_range = [-1,1]
add_idx = 1 #range内のどこのデータを表示するか指定します。
current_object = getattr(cfg, name)# getattr関数を使って、現在のオブジェクトを取得します。
setattr(current_object, 'context', time_range)# 時間的にどこからどこまでのデータを使うかを指定します。[-1,1]なら、現在のフレームと前後1フレームのデータを使います。

dataset = setup_dataset(cfg.dict[name])[0]
# display = DisplayDataset(dataset)
rgb, intrinsics = DisplayDataset.infer(dataset)

## フォルダを作成
if not os.path.exists(f"Infe_data/{name}"):
    os.mkdir(f"Infe_data/{name}")
os.chmod(f"Infe_data/{name}", 0o777)

## original画像をNumPy配列に変換して表示
# rgb_np = cv2.imread('/data/datasets/tiny/DDAD_tiny/000150/rgb_384_640/CAMERA_01/15616458296936490.jpg')
# print(rgb_np)
rgb_disp = rgb.squeeze().cpu().detach().numpy()
rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
# print(rgb_disp)
cv2.imwrite(f"Infe_data/{name}/rgb_input.png", rgb_disp)

## Infer from rgb and intrinsics
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
# pdb.set_trace()

# rgbサイズを384*640に変換し、それに合わせてintrinsicsも変換する関数
def resize_rgb_intrinsics(rgb, intrinsics):
    #rgbの幅と高さを取得
    rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
    #rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
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

## depth_predをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()

## depth_pred_npの最大値と最小値を取得
depth_pred_max = np.max(depth_pred_np)
depth_pred_min = np.min(depth_pred_np)

## depthをRGB画像に変換する関数
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
depth_rgb_image_vizdepth = viz_depth(depth_pred) 
print(depth_rgb_image_vizdepth.shape)   

## depthをRGBに変換
depth_rgb_image = depth_to_rgb(depth_pred_np, depth_pred_min, depth_pred_max)

## output（モノクロ）を保存
# cv2.imwrite("depth.png", depth_pred_np)
## RGBoutputを保存
cv2.imwrite(f"Infe_data/{name}/depth_color.png", depth_rgb_image)
#RGBoutput2を保存
write_image(f"Infe_data/{name}/depth_c_inv.png", depth_rgb_image_vizdepth)
print("最後まできたよ")