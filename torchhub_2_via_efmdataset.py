import torch
import numpy as np
import cv2 
import csv
import pdb
from efm_datasets.utils.config import read_config
from efm_datasets.utils.setup import setup_dataset
from efm_datasets.utils.data import make_batch, fold_batch, get_from_dict, interleave_dict, modrem
from efm_datasets.utils.viz import viz_depth,viz_inv_depth
from efm_datasets.utils.write import write_image
from scripts.display.DisplayDataset import DisplayDataset
import sys
import os
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

config = sys.argv[1]
name = sys.argv[2]
cfg = read_config(config)

## Override settings
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
    
    
    ##時短用
    # rgbサイズを384*640に変換し、それに合わせてintrinsicsも変換する関数
    def resize_rgb_intrinsics(rgb, intrinsics):
        #rgbの幅と高さを取得
        rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
        #rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
        resized_rgb = rgb.clone()
        resized_rgb = F.interpolate(rgb, size=(384, 640), mode='bilinear', align_corners=False)
        intrinsics2 = intrinsics.clone() 
        intrinsics2[0,0,0] = intrinsics[0,0,0].item()*640/rgb_w
        intrinsics2[0,0,2] = intrinsics[0,0,2].item()*640/rgb_w
        intrinsics2[0,1,1] = intrinsics[0,1,1].item()*384/rgb_h
        intrinsics2[0,1,2] = intrinsics[0,1,2].item()*384/rgb_h
        return resized_rgb, intrinsics2
    rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
    
    zerodepth_model = zerodepth_model.cuda()
    rgb2 = rgb2.cuda()
    intrinsics2 = intrinsics2.cuda()
    # infer via ZeroDepth
    depth_pred = zerodepth_model(rgb2, intrinsics2)#[1,3,384,640]
    depth_pred = depth_pred.cpu()
    
    depth_pred = torch.nn.functional.interpolate(depth_pred, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)#predをオリジナルサイズに変更
    print("depth_pred:",depth_pred.shape)
    print("depth_origin:",depth_origin.shape)
    
    
    
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
    print("loss:",loss)
    print("loss_mean:",loss.mean().item())
    print("loss_abs:",loss_abs)
    print("loss_abs_mean:",loss_abs.mean().item())
    
    ## depthをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
    depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()
    depth_origin_np = depth_origin.squeeze().cpu().detach().numpy()
    loss_np = loss.squeeze().cpu().detach().numpy()
    loss_abs_np = loss_abs.squeeze().cpu().detach().numpy()
    np.savez("temp_file.npz", depth_pred_np, depth_origin_np, loss_np, loss_abs_np, savepath, infe_camera, filename)

    # #時短用
    # loaded_data = np.load('temp_file.npz')
    # depth_pred_np = loaded_data['arr_0']
    # depth_origin_np = loaded_data['arr_1']
    # loss_np = loaded_data['arr_2']
    # loss_abs_np = loaded_data['arr_3']
    # savepath = loaded_data['arr_4']
    # infe_camera = loaded_data['arr_5']
    # filename = loaded_data['arr_6']
    
    #特定の値表示用
    mask_o10 = (loss_abs_np>=2).astype(np.float32)
    loss_abs_o10 = loss_abs_np * mask_o10 
    # viz_depthを使ってdepthを可視化
    depth_rgb_image_vizdepth = viz_depth(depth_pred_np, filter_zeros=True) 
    depth_origin_image_vizdepth = viz_depth(depth_origin_np, filter_zeros=True) 
    loss_image_vizdepth = viz_inv_depth(loss_np, filter_zeros=True)
    loss_abs_image_vizdepth = viz_inv_depth(loss_abs_np, filter_zeros=True)
    loss_abs_o10_image_vizdepth = viz_inv_depth(loss_abs_o10, filter_zeros=True)
    
    # # loss計算の関数を使ってみる
    # def compute_loss_and_metrics(pred_rgbs, gt_rgbs, pred_depths, gt_depths):
    #     """Compute loss and metrics given predictions and ground truth"""

    #     loss, metrics = [], {}

    #     # Calculate RGB losses
    #     if pred_rgbs is not None and 'rgb' in losses and weights[0] > 0.0:
    #         loss_rgb = []
    #         for pred, gt in zip(pred_rgbs, gt_rgbs):
    #             rgb_output = losses['rgb'](pred, gt)
    #             loss_rgb.append(weights[0] * rgb_output['loss'])
    #         loss.append(sum(loss_rgb) / len(loss_rgb))

    #     # Calculate depth losses
    #     if pred_depths is not None and 'depth' in losses and weights[1] > 0.0:
    #         loss_depth = []
    #         for pred, gt in zip(pred_depths, gt_depths):
    #             depth_output = losses['depth'](pred, gt)
    #             loss_depth.append(weights[1] * depth_output['loss'])
    #         loss.append(sum(loss_depth) / len(loss_depth))

    #     if len(loss) == 2 and scale_loss:
    #         ratio_rgb_depth = loss[1].item() / loss[0].item()
    #         loss[0] = loss[0] * ratio_rgb_depth

    #     loss = sum(loss) / len(loss)

    #     return loss, metrics
    
    # loss_hoge, metrics_hoge = compute_loss_and_metrics(None, None, depth_pred_np, depth_origin_np)
    # print(loss_hoge)
    # print(metrics_hoge)
    
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
    write_image(f"{savepath}inf_result/loss_abs{infe_camera}_{filename}.png", loss_abs_image_vizdepth)
    write_image(f"{savepath}inf_result/loss_abs_o2_{infe_camera}_{filename}.png", loss_abs_o10_image_vizdepth)
    
    #データをint型で1次元に変換
    depth_pred_np = depth_pred_np.astype(np.int32).flatten()
    depth_origin_np = depth_origin_np.astype(np.int32).flatten()
    loss_np = loss_np.astype(np.int32).flatten()
    loss_abs_np = loss_abs_np.astype(np.int32).flatten()
    
    #結果をヒストグラムでプロット
    from graph import plot_distribution
    plot_distribution(depth_origin_np, depth_pred_np, loss_np, loss_abs_np, savepath, infe_camera, filename)
    
print("最後まできたよ")

