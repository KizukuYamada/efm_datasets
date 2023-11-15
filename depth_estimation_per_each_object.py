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
from YOLOv8.ultralytics import YOLO

data_type = "dataset"#VIDEO,dataset,IMAGE
img_file = 'YOLOv8/examples/bus.jpg'
#TMC動画に対する処理
video_path = "YOLOv8/examples/20210916_084834000_iOS_short.mp4"
cam_para = [[1.20866229e+03,0.00000000e+00,6.33281682e+02],
            [0.00000000e+00,1.20860740e+03,3.63294680e+02], 
            [0.00000000e+00,0.00000000e+00,1.00000000e+00]]

# hoge=cv2.imread(img_file)
# フレームを読み込む

YOLO_conf = 0.3
# pdb.set_trace()
if data_type == "dataset":
    config = sys.argv[1]
    name = sys.argv[2]
    cfg = read_config(config)

    ## Override settings
    time_range = [-2,2]
    infe_camera = 0 #どのカメラのデータを表示するか指定します。0なら左、1なら右です。
    infe_range = [0,0]
    # 解析範囲がtime_rangeを超えていたら、time_rangeに合わせる
    infe_range[0] = max(time_range[0],infe_range[0])
    infe_range[1] = min(time_range[1],infe_range[1])

elif data_type == "VIDEO":
    cap = cv2.VideoCapture(video_path)
    # pdb.set_trace()
    fm_max =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("width:",width)
    print("height:",height)
    flame_num = [300,305]   
    # flame_num = [0,fm_max]   
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    
    infe_range = [flame_num[0],flame_num[1]]
    infe_camera = 99
    filepath = video_path
    filename = os.path.splitext(os.path.basename(filepath))
    # pdb.set_trace()
    savepath = filepath.split(".")[0]
    out = cv2.VideoWriter(f"{savepath}inf_result/output_video2.mp4", fourcc, fps, (width, height))
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
    device = torch.device(f"cuda:{remainder}" if torch.cuda.is_available() else "cpu")
    
    if data_type == "dataset":
        add_idx = i #range内のどこのデータを表示するか指定します。
        dataset = setup_dataset(cfg.dict[name])[0]
        rgb, intrinsics, filepath, filename, depth_origin = DisplayDataset.infer(dataset,add_idx,infe_camera)#rgb:torch.Size([1, 3, 400, 879]),depth_origin:torch.Size([1, 1, 400, 879])
        savepath = filepath.split("image_")[0]
        rgb_disp = rgb.squeeze().cpu().detach().numpy()
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)#(400, 879, 3)
    elif data_type == "VIDEO":
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()#frame:(720, 1280, 3)
        # cv2.imshow("test",frame)
        # key = cv2.waitKey(1)
        # pdb.set_trace()
        intrinsics = torch.tensor(cam_para).unsqueeze(0)
        rgb = torch.tensor(frame).permute(2,0,1).unsqueeze(0)/255. #True:torch.Size([1, 3, 400, 879]),False:torch.Size([1, 3, 384, 640]),torch.float32
        depth_origin = rgb.mean(dim=1, keepdim=True)#torch.Size([1, 1, 384, 640])
        rgb_disp = frame
    else:
        intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0) #True:torch.Size([1, 3, 3])
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)#True:torch.float32
        rgb = torch.tensor(cv2.imread(img_file)).permute(2,0,1).unsqueeze(0)/255. #True:torch.Size([1, 3, 400, 879]),False:torch.Size([1, 3, 384, 640]),torch.float32
        filepath = img_file
        filename = os.path.splitext(os.path.basename(filepath))
        savepath = filepath.split(".")[0]
        depth_origin = rgb.mean(dim=1, keepdim=True)#torch.Size([1, 1, 384, 640])
        rgb_disp = cv2.imread(img_file)


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
    zerodepth_model.to(device)
    rgb2 = rgb2.to(device)#True:torch.Size([1, 3, 384, 640]) False:
    intrinsics2 = intrinsics2.to(device)#True:torch.Size([1, 3, 3])

    # infer via ZeroDepth
    depth_pred = zerodepth_model(rgb2, intrinsics2)#True:torch.Size([1, 1, 384, 640]) False:torch.Size([1, 1, 384, 640])
    depth_pred = depth_pred.cpu()#True:torch.Size([1, 1, 384, 640]) False:
    pdb.set_trace()
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
    # print("loss:",loss)
    # print("loss_mean:",loss.mean().item())
    # print("loss_abs:",loss_abs)
    # print("loss_abs_mean:",loss_abs.mean().item())
    
    ## depthをNumPy配列に変換して勾配情報を切り離して、CPUに送って、次元削減
    depth_pred_np = depth_pred.squeeze().detach().numpy()#True:(400, 879) False:(384, 640)
    depth_origin_np = depth_origin.squeeze().cpu().detach().numpy()
    loss_np = loss.squeeze().cpu().detach().numpy()
    loss_abs_np = loss_abs.squeeze().cpu().detach().numpy()
    # np.savez("temp_file.npz", depth_pred_np, depth_origin_np, loss_np, loss_abs_np, savepath, infe_camera, filename)

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

    
    ## YOLOv8推論
    seg_model = YOLO('yolov8n-seg.pt')
    seg_model.to(device)
    
    rgb_disp_gpu = torch.tensor(rgb_disp).to(device)

    seg_results = seg_model(rgb_disp,show=True,conf=YOLO_conf, save=False,save_txt=False,project=f"{savepath}inf_result/YOLO_seg{infe_camera}_{filename}")
    #retina_masks=True,
    seg_result = seg_results[0]

    # pdb.set_trace()
    # print("seg_result:",seg_result_box.device)
    ## seg_result.masks.dataのshapeをtorch.Size([7, 320, 640])からtorch.Size([7, 400, 879])に変更
    if seg_result.masks is not None:
        seg_result_masks_gpu = seg_result.masks.data.to(device) #torch.Size([3, 384, 640])
    
        # ターゲットのサイズ
        target_height = seg_result.orig_shape[0]
        target_width = seg_result.orig_shape[1]
        # ニアレストネイバー補間を使ってリサイズ
        resized_masks = F.interpolate(seg_result_masks_gpu.unsqueeze(0),  # バッチ次元を追加
                                    size=(target_height, target_width),   # 新しいサイズを指定
                                    mode='nearest').squeeze(0)            # バッチ次元を削除
        
        ## depth_predのセグメンテーション範囲のみを抽出
        # pdb.set_trace()
        depth_pred = depth_pred.to(device)
        depth_map = depth_pred.squeeze()  # 結果はtorch.Size([400, 879])
        depth_true_map = depth_origin.to(device).squeeze() #torch.Size([400, 879])
        # Depthをマスクの数だけチャネル方向に複製します。
        depth_maps_expanded = depth_map.expand_as(resized_masks)#torch.Size([7,400, 879])
        depth_true_maps_expanded = depth_true_map.expand_as(resized_masks)#torch.Size([7,400, 879])
        
        
        
        # マスクを適用して、マスクされた領域の予測Depth値と正解Depth値を取得
        # pdb.set_trace()
        masked_depth = depth_maps_expanded * resized_masks
        masked_true_depth = depth_true_maps_expanded * resized_masks
        
        #マスク範囲を表示
        for l in range(resized_masks.size(0)):
            masked_depth_np = viz_depth(masked_depth[l], filter_zeros=True) 
            masked_true_depth_np = viz_depth(masked_true_depth[l], filter_zeros=True) 
            # depth_map_np = masked_depth[0].cpu().detach().numpy()
            # masked_true_depth_np = masked_true_depth[0].cpu().detach().numpy()
            # # データのスケールを調整（オプション）
            # # 例えば、最小値を0、最大値を255にスケーリング
            # masked_true_depth_np = (masked_true_depth_np - masked_true_depth_np.min()) / (masked_true_depth_np.max() - masked_true_depth_np.min()) * 255
            # masked_true_depth_np = masked_true_depth_np.astype(np.uint8)
            # pdb.set_trace()
            # 画像を表示
            cv2.imshow(f'Depth_True Map{l}', masked_true_depth_np)
            cv2.imshow(f'Depth Map{l}', masked_depth_np)
            key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # pdb.set_trace()
        ## 検出したモノの数だけループ
        ave_obj_depth = torch.zeros(masked_depth.size(0))
        ave_obj_true_depth = torch.zeros(masked_depth.size(0))
        posi_x = torch.zeros(masked_depth.size(0))
        posi_y = torch.zeros(masked_depth.size(0))
        posi = torch.zeros(masked_depth.size(0),2)
        # pdb.set_trace()
        # rgb_text = rgb_disp.copy()#numpy.ndarray#(720, 1280, 3)
        #セグメンテーション画像上にDepthを追記
        # pdb.set_trace()
        rgb_text = seg_result.plot().copy()#(720, 1280, 3)
        # rgb_text = rgb_text.cpu()#(720, 1280, 3)
        rgb_true_text = rgb_disp.copy()
        for j in range(masked_depth.size(0)):
            obj_depth = masked_depth[j]#予測値
            obj_true_depth = masked_true_depth[j]#正解値 torch.Size([400, 879])
            # Depth値が0の部分を無視するために、マスクを適用
            # 0より値の大きいインデックスを取得します。
            over_zero_indices = obj_depth > 0
            # 0でない値だけを抽出します。
            if over_zero_indices.any():
                # 0でない値だけを抽出して平均値を計算します。
                ave_obj_depth[j] = torch.mean(obj_depth[obj_depth > 0])
                ave_obj_true_depth[j] = torch.mean(obj_true_depth[obj_true_depth > 0])
            else:
                # 0でない値がない場合は、平均値を定義することができません。
                # ave_obj_depth[j] = 0  # 既に0で初期化されているか、適切な
                # 値を設定してください。
                pass  # 何もしない
        
            #位置計算
            posi_x[j] = (seg_result.boxes.xyxy[j][0]+seg_result.boxes.xyxy[j][2])/2
            posi_y[j] = (seg_result.boxes.xyxy[j][1]+seg_result.boxes.xyxy[j][3])/2
            posi[j] = torch.tensor([posi_x[j],posi_y[j]])
            
            
            ## Depthを画像上の位置に表示
            # テキスト設定
            # pdb.set_trace()
            ave_obj_depth_text = round(ave_obj_depth[j].item(),1)
            ave_obj_true_depth_text = round(ave_obj_true_depth[j].item(),1)
            pred_text = f"{ave_obj_depth_text}"  # 画像に挿入するテキスト
            true_text = f"{ave_obj_true_depth_text}"  # 画像に挿入するテキスト
            org = (int(posi_x[j]),int(posi_y[j]))  # テキストを挿入する位置（左下の座標）
            # pdb.set_trace()
            font = cv2.FONT_HERSHEY_SIMPLEX  # フォントタイプ
            fontScale = 1  # フォントサイズ
            color_b = (255, 0, 0)  # テキストの色（BGR）
            color_r = (0, 0, 255)  # テキストの色（BGR）
            thickness = 2  # テキストの太さ
            
            # テキストを画像に挿入
            cv2.putText(rgb_text, pred_text, org, font, fontScale, color_b, thickness, cv2.LINE_AA)
            cv2.putText(rgb_true_text, true_text, org, font, fontScale, color_r, thickness, cv2.LINE_AA)    
    else:
        rgb_text = rgb_disp.copy()
        rgb_true_text = rgb_disp.copy()
        print("物体検出なし")
        
    # rgb_text_cpu = rgb_text.cpu().detach().numpy()
    print("rgb_text:",rgb_text.shape)#(720, 1280, 3)
    ## フォルダを作成
    if not os.path.exists(f"{savepath}inf_result"):
        os.mkdir(f"{savepath}inf_result")
    os.chmod(f"{savepath}inf_result", 0o777)
    if not os.path.exists(f"{savepath}inf_result/hoge"):
        os.mkdir(f"{savepath}inf_result/hoge")
    os.chmod(f"{savepath}inf_result", 0o777)

    ## input（RGB）を保存
    # cv2.imwrite(f"{savepath}inf_result/rgb_input{infe_camera}_{filename[0]}.png", rgb_disp)
    cv2.imwrite(f"{savepath}inf_result/hoge/rgb_text{infe_camera}_{filename[0]}_{i}.png", rgb_text)
    # cv2.imwrite(f"{savepath}inf_result/rgb_true_text{infe_camera}_{filename[0]}.png", rgb_true_text)
    # depthのoutput（カラー、TRIバージョン）を保存
    write_image(f"{savepath}inf_result/depth_c_inv{infe_camera}_{filename[0]}.png", depth_rgb_image_vizdepth)
    write_image(f"{savepath}inf_result/depth_true_inv{infe_camera}_{filename[0]}.png", depth_origin_image_vizdepth)
    # write_image(f"{savepath}inf_result/loss_{infe_camera}_{filename[0]}.png", loss_image_vizdepth)
    write_image(f"{savepath}inf_result/loss_abs{infe_camera}_{filename[0]}.png", loss_abs_image_vizdepth)
    # write_image(f"{savepath}inf_result/loss_abs_o2_{infe_camera}_{filename}.png", loss_abs_o10_image_vizdepth)#Depthの誤差がXよりも大きいところだけ保存
    ##動画として保存
    # pdb.set_trace()
    if data_type == "VIDEO":
        if ret:
            # cv2.putText(rgb_text, 'Frame Number: {}'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))),
            #         (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 処理済みのフレームを出力動画に追加
            # cv2.imshow("test",rgb_text)
            # key = cv2.waitKey(1)
            out.write(rgb_text)
        else:
            print(f"{i}フレーム目、読み込み失敗")
    #データをint型で1次元に変換
    depth_pred_np = depth_pred_np.astype(np.int32).flatten()
    depth_origin_np = depth_origin_np.astype(np.int32).flatten()
    loss_np = loss_np.astype(np.int32).flatten()
    loss_abs_np = loss_abs_np.astype(np.int32).flatten()
    
    
    if seg_result.masks is not None:
        #結果をヒストグラムでプロット
        from graph import plot_distribution, compare_distribution
        # pdb.set_trace()
        compare_distribution(masked_true_depth, masked_depth, savepath, infe_camera, filename)
        # print('ave_obj_depth:',ave_obj_depth)
        # print('ave_obj_ture_depth:',ave_obj_true_depth)
        # plot_distribution(depth_origin_np, depth_pred_np, loss_np, loss_abs_np, savepath, infe_camera, filename)
    
    
    del depth_pred,rgb,rgb2,intrinsics,intrinsics2,depth_pred_np,depth_origin_np,loss_np,loss_abs_np
    del seg_results
    if seg_result.masks is not None:
        del seg_result_masks_gpu, seg_model
        del depth_map, depth_maps_expanded, masked_depth, ave_obj_depth, posi_x, posi_y, posi, rgb_text,resized_masks
    # del seg_results, depth_map, depth_maps_expanded, masked_depth, ave_obj_depth, posi_x, posi_y, posi, rgb_text
    # del seg_model
    torch.cuda.empty_cache()
    print(f"データ{filename[0]}保存したよ")
    # if data_type == "VIDEO":
    #     out.release()
    
    
if data_type == "VIDEO":
    out.release()    
    cap.release()
print("最後まできたよ")

