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
# from YOLOv8.examples.yolo_seg import yolo_seg
from YOLOv8.ultralytics import YOLO

dataloader_use = False


if dataloader_use:
    config = sys.argv[1]
    name = sys.argv[2]
    cfg = read_config(config)

    ## Override settings
    time_range = [-2,2]
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
else:
    infe_range = [0,0]
    infe_camera = 99

# zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo="check")

# if torch.cuda.device_count() > 1:
#     zerodepth_model = torch.nn.DataParallel(zerodepth_model)
# zerodepth_model.to('cuda')
# zerodepth_model = zerodepth_model.cuda()
for i in range(infe_range[0],infe_range[1]+1):
    # i/2の余りを計算
    remainder = i%3
    remainder = 0        
    device = torch.device(f"cuda:{remainder}" if torch.cuda.is_available() else "cpu")
        
    add_idx = i #range内のどこのデータを表示するか指定します。
    # current_object = getattr(cfg, name)# getattr関数を使って、現在のオブジェクトを取得します。
    # setattr(current_object, 'context', time_range)# 時間的にどこからどこまでのデータを使うかを指定します。[-1,1]なら、現在のフレームと前後1フレームのデータを使います。
    if dataloader_use:
        dataset = setup_dataset(cfg.dict[name])[0]
        rgb, intrinsics, filepath, filename, depth_origin = DisplayDataset.infer(dataset,add_idx,infe_camera)#rgb:torch.Size([1, 3, 400, 879]),depth_origin:torch.Size([1, 1, 400, 879])
        savepath = filepath.split("image_")[0]
        rgb_disp = rgb.squeeze().cpu().detach().numpy()
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)#(400, 879, 3)
    else:
        img_file = 'examples/ddad_sample_new.png'
        intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0) #True:torch.Size([1, 3, 3])
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)#True:torch.float32
        rgb = torch.tensor(cv2.imread(img_file)).permute(2,0,1).unsqueeze(0)/255. #True:torch.Size([1, 3, 400, 879]),False:torch.Size([1, 3, 384, 640]),torch.float32
        filepath = img_file
        filename = os.path.splitext(os.path.basename(filepath))
        savepath = filepath.split("image_")[0]
        depth_origin = rgb.mean(dim=1, keepdim=True)#torch.Size([1, 1, 384, 640])
        rgb_disp = cv2.imread(img_file)
        # rgb_disp = rgb.squeeze().cpu().detach().numpy()#
        # rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        # rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)#True:(384, 640, 3)
        
    
    # # depthを取得
    
    ## original画像をNumPy配列に変換して表示
    
    

    
    
    
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
    
    if rgb is None:
        print("Error: 'rgb' is None")
    elif intrinsics is None:
        print("Error: 'intrinsics' is None")
    else:   
        rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
        
    zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo="check")
    # if torch.cuda.device_count() > 1:
    #     zerodepth_model = torch.nn.DataParallel(zerodepth_model)
    zerodepth_model.to(device)
    rgb2 = rgb2.to(device)#True:torch.Size([1, 3, 384, 640]) False:
    intrinsics2 = intrinsics2.to(device)#True:torch.Size([1, 3, 3])
    # pdb.set_trace()
    # infer via ZeroDepth
    depth_pred = zerodepth_model(rgb2, intrinsics2)#True:torch.Size([1, 1, 384, 640]) False:torch.Size([1, 1, 384, 640])
    # pdb.set_trace()
    depth_pred = depth_pred.cpu()#True:torch.Size([1, 1, 384, 640]) False:
    # zerodepth_model = zerodepth_model.cpu()
    # del zerodepth_model
    
    depth_pred = torch.nn.functional.interpolate(depth_pred, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)#predをオリジナルサイズに変更
    print("depth_pred:",depth_pred.shape)#True:torch.Size([1, 1, 400, 879]) False:torch.Size([1, 1, 384, 640])
    print("depth_origin:",depth_origin.shape)#True:torch.Size([1, 1, 400, 879]) False:torch.Size([1, 1, 384, 640])
    
    
    
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
    depth_pred_np = depth_pred.squeeze().detach().numpy()#True:(400, 879) False:(384, 640)
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
    

    
    ## YOLOv8推論
    # yolo_result = yolo_seg(rgb)
    seg_model = YOLO('yolov8n-seg.pt')
    # seg_results = seg_model('YOLOv8/examples/bus.jpg',show=True, save=False,save_txt=False)#(810,1100)OK
    # seg_results = seg_model('YOLOv8/examples/rgb_input0_0000000005.png',show=True, save=True,save_txt=False)#
    # pdb.set_trace()

    seg_results = seg_model(rgb_disp,show=True, conf=0.8, save=False,save_txt=False)
    seg_result = seg_results[0]
    
    pdb.set_trace()
    ## seg_result.masks.dataのshapeをtorch.Size([7, 320, 640])からtorch.Size([7, 400, 879])に変更
    if seg_result.masks is not None:
        seg_result_masks = seg_result.masks.data #torch.Size([3, 384, 640])
    
    
        # ターゲットのサイズ
        target_height = seg_result.orig_shape[0]
        target_width = seg_result.orig_shape[1]
        # ニアレストネイバー補間を使ってリサイズ
        resized_masks = F.interpolate(seg_result.masks.data.unsqueeze(0),  # バッチ次元を追加
                                    size=(target_height, target_width),   # 新しいサイズを指定
                                    mode='nearest').squeeze(0)            # バッチ次元を削除
        
        ## depth_predのセグメンテーション範囲のみを抽出
        depth_pred = depth_pred.cuda()
        depth_map = depth_pred.squeeze()  # 結果はtorch.Size([400, 879])
        # Depthをマスクの数だけチャネル方向に複製します。
        depth_maps_expanded = depth_map.expand_as(resized_masks)#torch.Size([7,400, 879])
        
        # マスクを適用して、マスクされた領域のDepth値のみを取得
        masked_depth = depth_maps_expanded * resized_masks
        ## 検出したモノの数だけループ
        ave_obj_depth = torch.zeros(masked_depth.size(0))
        posi_x = torch.zeros(masked_depth.size(0))
        posi_y = torch.zeros(masked_depth.size(0))
        posi = torch.zeros(masked_depth.size(0),2)
        rgb_text = rgb_disp.copy()
        for j in range(masked_depth.size(0)):
            obj_depth = masked_depth[j]
            # Depth値が0の部分を無視するために、マスクを適用
            # 0でない値のインデックスを取得します。
            non_zero_indices = obj_depth != 0
            # 0でない値だけを抽出します。
            if non_zero_indices.any():
                # 0でない値だけを抽出して平均値を計算します。
                ave_obj_depth[j] = torch.mean(obj_depth[non_zero_indices])
            else:
                # 0でない値がない場合は、平均値を定義することができません。
                # ave_obj_depth[j] = 0  # 既に0で初期化されているか、適切な値を設定してください。
                pass  # 何もしない
        
            #位置計算
            posi_x[j] = (seg_result.boxes.xyxy[j][0]+seg_result.boxes.xyxy[j][2])/2
            posi_y[j] = (seg_result.boxes.xyxy[j][1]+seg_result.boxes.xyxy[j][3])/2
            posi[j] = torch.tensor([posi_x[j],posi_y[j]])
            
            
            ## Depthを画像上の位置に表示
            # テキスト設定
            # pdb.set_trace()
            ave_obj_depth_text = round(ave_obj_depth[j].item(),1)
            text = f"{ave_obj_depth_text}"  # 画像に挿入するテキスト
            org = (int(posi_x[j]),int(posi_y[j]))  # テキストを挿入する位置（左下の座標）
            # pdb.set_trace()
            font = cv2.FONT_HERSHEY_SIMPLEX  # フォントタイプ
            fontScale = 1  # フォントサイズ
            color = (255, 0, 0)  # テキストの色（BGR）
            thickness = 2  # テキストの太さ
            
            # テキストを画像に挿入
            cv2.putText(rgb_text, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        
    else:
        rgb_text = rgb_disp.copy()
        print("物体検出なし")
    
    ## フォルダを作成
    if not os.path.exists(f"{savepath}inf_result"):
        os.mkdir(f"{savepath}inf_result")
    os.chmod(f"{savepath}inf_result", 0o777)

    ## input（RGB）を保存
    cv2.imwrite(f"{savepath}inf_result/rgb_input{infe_camera}_{filename}.png", rgb_disp)
    cv2.imwrite(f"{savepath}inf_result/rgb_text{infe_camera}_{filename}.png", rgb_text)
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
    # from graph import plot_distribution
    # plot_distribution(depth_origin_np, depth_pred_np, loss_np, loss_abs_np, savepath, infe_camera, filename)
    
    # for namei, param in seg_model.named_parameters():
        # print(namei, param.device)
    # pdb.set_trace()
    
    del depth_pred,rgb,rgb2,intrinsics,intrinsics2,depth_pred_np,depth_origin_np,loss_np,loss_abs_np
    del seg_results
    # del seg_results, depth_map, depth_maps_expanded, masked_depth, ave_obj_depth, posi_x, posi_y, posi, rgb_text
    del seg_model
    torch.cuda.empty_cache()
    
    
    
    
print("最後まできたよ")

