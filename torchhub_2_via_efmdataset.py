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
import torch.nn.functional as F

config = sys.argv[1]
print(config)
name = sys.argv[2]
cfg = read_config(config)

## 設定を上書き
time_range = [-3,3]
infe_camera = 0 #どのカメラのデータを表示するか指定します。0なら左、1なら右です。
infe_range = [0,0]
# 解析範囲がtime_rangeを超えていたら、time_rangeに合わせる
if time_range[0] <= infe_range[0]:
    pass
else:
    infe_range[0] = time_range[0]
if infe_range[1] <= time_range[1]:
    pass
else:
    infe_range[1] = time_range[1]

for i in range(infe_range[0],infe_range[1]+1):
    add_idx = i #range内のどこのデータを表示するか指定します。
    current_object = getattr(cfg, name)# getattr関数を使って、現在のオブジェクトを取得します。
    setattr(current_object, 'context', time_range)# 時間的にどこからどこまでのデータを使うかを指定します。[-1,1]なら、現在のフレームと前後1フレームのデータを使います。

    dataset = setup_dataset(cfg.dict[name])[0]
    # display = DisplayDataset(dataset)
    rgb, intrinsics, filepath, filename, depth_origin = DisplayDataset.infer(dataset,add_idx,infe_camera)
    # pdb.set_trace()
    # # depthを取得
    # depth = dataset.get_depth(add_idx, infe_camera)
    savepath = filepath.split("image_")[0]
    ## original画像をNumPy配列に変換して表示
    rgb_disp = rgb.squeeze().cpu().detach().numpy()
    rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
    rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)

    ## Infer from rgb and intrinsics
    zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
    
    # rgbサイズを384*640に変換し、それに合わせてintrinsicsも変換する関数
    def resize_rgb_intrinsics(rgb, intrinsics):
        #rgbの幅と高さを取得
        rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
        #rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
        resized_rgb = rgb.clone()
        resized_rgb = F.interpolate(rgb, size=(384, 640), mode='bilinear', align_corners=False)
        # print(rgb.shape)
        intrinsics2 = intrinsics.clone() 
        intrinsics2[0,0,0] = intrinsics[0,0,0].item()*640/rgb_w
        intrinsics2[0,0,2] = intrinsics[0,0,2].item()*640/rgb_w
        intrinsics2[0,1,1] = intrinsics[0,1,1].item()*384/rgb_h
        intrinsics2[0,1,2] = intrinsics[0,1,2].item()*384/rgb_h
        # print("intrinsics2:", intrinsics2)
        # print("intrinsics:", intrinsics)
        return resized_rgb, intrinsics2

    rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
    depth_pred = zerodepth_model(rgb2, intrinsics2)#[1,3,384,640]
    depth_pred = torch.nn.functional.interpolate(depth_pred, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)#predをオリジナルサイズに変更
    # depth_origin2 = torch.nn.functional.interpolate(depth_origin, size=(384, 640), mode='bilinear', align_corners=False)#[1,3,384,640]に変更
    # print("depth_pred:",depth_pred.shape)
    # print("depth_origin:",depth_origin.shape)
    print("depth_pred:",depth_pred.shape)
    print("depth_origin:",depth_origin.shape)
    # pdb.set_trace()
    # 正解Depthと予測Deothの誤差をhuber関数で比較
    def masked_huber_loss(pred, target, threshold=0, delta=1.0):
        mask = (target > threshold).float()
        # huber lossを計算
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=delta)
        # 単純な値の引き算(絶対値)を計算
        loss_abs = torch.abs(pred - target)
        # マスク(正解点群のあるピクセルだけ誤差計算)を適用
        masked_loss = loss * mask
        masked_loss_abs = loss_abs * mask
        return masked_loss, masked_loss_abs

    loss, loss_abs = masked_huber_loss(depth_pred, depth_origin)
    # pdb.set_trace()
    print("loss:",loss)
    print("loss_mean:",loss.mean().item())
    print("loss_abs:",loss_abs)
    print("loss_abs_mean:",loss_abs.mean().item())
    
    ## depthをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
    depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()
    depth_origin_np = depth_origin.squeeze().cpu().detach().numpy()
    loss_np = loss.squeeze().cpu().detach().numpy()

    # viz_depthを使ってdepthを可視化
    depth_rgb_image_vizdepth = viz_depth(depth_pred, filter_zeros=True) 
    depth_origin_image_vizdepth = viz_depth(depth_origin, filter_zeros=True) 
    loss_image_vizdepth = viz_depth(loss, filter_zeros=True)
    
    ## フォルダを作成
    if not os.path.exists(f"{savepath}inf_result"):
        os.mkdir(f"{savepath}inf_result")
    os.chmod(f"{savepath}inf_result", 0o777)

    ## input（RGB）を保存
    cv2.imwrite(f"{savepath}inf_result/rgb_input{infe_camera}_{filename}.png", rgb_disp)
    ## depthのoutput（モノクロ）を保存
    # cv2.imwrite("depth{infe_camera}_{add_idx}.png", depth_pred_np)
    # depthのoutput（カラー、TRIバージョン）を保存
    write_image(f"{savepath}inf_result/depth_c_inv{infe_camera}_{filename}.png", depth_rgb_image_vizdepth)
    write_image(f"{savepath}inf_result/depth_true_inv{infe_camera}_{filename}.png", depth_origin_image_vizdepth)
    write_image(f"{savepath}inf_result/loss_{infe_camera}_{filename}.png", loss_image_vizdepth)
print("最後まできたよ")