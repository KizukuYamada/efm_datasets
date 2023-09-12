import torch
import numpy as np
import cv2 
import csv
import pdb
from efm_datasets.utils.config import read_config
from efm_datasets.utils.setup import setup_dataset
from scripts.display.DisplayDataset import DisplayDataset

import sys
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)

# intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)
# rgb = torch.tensor(cv2.imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.

# intrinsics = torch.tensor(np.load('/mnt/fsx/datasets/tiny/DDAD_tiny/000150/intrinsics/CAMERA_01/15616458296936490.npy')).unsqueeze(0)
# rgb = torch.tensor(cv2.imread('/mnt/fsx/datasets/tiny/DDAD_tiny/000150/rgb/CAMERA_01/15616458296936490.png')).permute(2,0,1).unsqueeze(0)/255.
config = sys.argv[1] 
print(config)
name = sys.argv[2]

cfg = read_config(config)
dataset = setup_dataset(cfg.dict[name])[0]
# pdb.set_trace()
display = DisplayDataset(dataset)
# display.loop()
# intrinsics = torch.tensor(np.load('/mnt/fsx/datasets/tiny/DDAD_tiny/000150/intrinsics/CAMERA_01/15616458296936490.npy')).unsqueeze(0)
# rgb = torch.tensor(cv2.imread('/mnt/fsx/datasets/tiny/DDAD_tiny/000150/rgb/CAMERA_01/15616458296936490.png')).permute(2,0,1).unsqueeze(0)/255.
# pdb.set_trace()

# original画像をNumPy配列に変換して表示
rgb_np = cv2.imread('/mnt/fsx/datasets/tiny/DDAD_tiny/000150/rgb/CAMERA_01/15616458296936490.png')
# rgb.squeeze().cpu().detach().numpy()
# cv2.imwrite("rgb.png", rgb_np)

depth_pred = zerodepth_model(rgb, intrinsics)
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

# depthをRGBに変換
depth_rgb_image = depth_to_rgb(depth_pred_np, depth_pred_min, depth_pred_max)

#outputを保存
# cv2.imwrite("depth.png", depth_pred_np)
#RGBoutputを保存
cv2.imwrite("depth_color.png", depth_rgb_image)
