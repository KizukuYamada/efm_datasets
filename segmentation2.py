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
from graph import mask_disp, compare_distribution, compare_true_pred, plot_distribution
from functions import remove_outliers
import datetime
import subprocess

data_type = "dataset" #VIDEO,dataset,IMAGE
img_file = 'YOLOv8/examples/bus.jpg'
img_file = 'mask/masked_images/rgb_input99_20210916_084834000_iOS_short_UUmask.png'
#VIDEOに対する前設定
video_path = "/data/datasets/Platooning/20210916_084834000_iOS_short.mp4"
video_path = "/data/datasets/Platooning/20210916_084834000_iOS.mov"
video_path = "/data/datasets/Platooning/20231030/20231030/nobori/20231030_065630000_iOS.MOV"
cam_para = [[1.20866229e+03,0.00000000e+00,6.33281682e+02],
            [0.00000000e+00,1.20860740e+03,3.63294680e+02], 
            [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
# cam_para = [[3.2704918e+03,0.00000000e+00,6.33281682e+02],
#             [0.00000000e+00,3.2704918e+03,3.63294680e+02], 
#             [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
#空箱の定義
ave_obj_true_depth_all = []
ave_obj_depth_all = []

#YOLOの設定
seg_model = YOLO('yolov8n-seg.pt')
# YOLO_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_device = torch.device("cpu")
seg_model.to(YOLO_device)
YOLO_conf = 0.20

if data_type == "dataset":
    config = sys.argv[1]
    name = sys.argv[2]  
    cfg = read_config(config)
    #splitに何枚分のデータが入っているかを確認
    sp_path = f"{cfg.dict[name].path[0]}/{cfg.dict[name].split[0]}"
    data_max = 0
    with open(sp_path, 'r') as file:
        for line in file:
            data_max += 1
    print("data_max:",data_max)
    #cfg設定の上書き
    cfg.dict[name].context[0]=0
    cfg.dict[name].context[1]=data_max-1
    # cfg.dict[name].context[1]=6
    print(cfg.dict[name])

    ## Override settings
    infe_camera = 0 #どのカメラのデータを表示するか指定します。0なら左、1なら右です。
    infe_range = [0,4]
    # infe_range = [460,463]
    step = 1
    # pdb.set_trace()
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    dataset = setup_dataset(cfg.dict[name])[0]
    data = dataset[0]
    data = make_batch(data)
    data = fold_batch(data)
    filepath0 = data["filename"][((0, infe_camera))]
    savepath = filepath0.split("image_")[0]
    # pdb.set_trace()
    st_data = data["filename"][((infe_range[0], infe_camera))].split("/")[-1].split(".")[0]
    end_data = data["filename"][((infe_range[1], infe_camera))].split("/")[-1].split(".")[0]
    fps = 1
    if name == "DrivingStereo":
        width = 879
        height = 400
        timestamp_file = os.path.join(filepath0.split("2018-10-11-17-08-31hoge/2018-10-11-17-08-31hoge")[0], "timestamp.txt")
        with open(timestamp_file, 'r') as f:
            timestamps = [datetime.datetime.strptime(line.strip().split('_')[1][:-4], '%Y-%m-%d-%H-%M-%S-%f') for line in f]
        durations = [(timestamps[i] - timestamps[i-1]).total_seconds()*1000 for i in range(1, len(timestamps))]+[1000]
        # pdb.set_trace()
    elif name == "KITTI":
        width = 1242
        height = 375
        # pdb.set_trace()
        timestamp_file = os.path.join(filepath0.split("image_02")[0], "oxts/timestamps.txt")
        with open(timestamp_file, 'r') as f:
            timestamps = [datetime.datetime.strptime(line[:26].strip(), '%Y-%m-%d %H:%M:%S.%f') for line in f]
            # pdb.set_trace()
        durations = [(timestamps[i] - timestamps[i-1]).total_seconds()*1000 for i in range(1, len(timestamps))]+[1000]
    else:
        width = 879
        height = 400
    # pdb.set_trace()
    videooutpath = f"{savepath}inf_result/hoge/{name}_{infe_camera}_{st_data}_{end_data}_Y{YOLO_conf}.mp4"
    out = cv2.VideoWriter(videooutpath, fourcc, fps/step, (width, height))
    out2 = cv2.VideoWriter(f"{savepath}inf_result/hoge/depth_{name}_{infe_camera}_{st_data}_{end_data}_Y{YOLO_conf}.mp4", fourcc, fps/step, (width, height))
elif data_type == "VIDEO":
    cap = cv2.VideoCapture(video_path)
    # pdb.set_trace()
    fm_max =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, x2 = 300, width-500
    y1, y2 = 250, 700
    # x1, x2 = 0, width
    # y1, y2 = 0, height
    width = x2-x1
    height = y2-y1
    print("width:",width)
    print("height:",height)
    # flame_num = [1,fm_max]   
    flame_num = [30000,31000]   
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter.fourcc(*'h264')
    infe_range = [flame_num[0],flame_num[1]]
    step = int(fps/1) #0.5秒ごとになるように設定
    infe_camera = 99
    filepath = video_path
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # pdb.set_trace()
    savepath = filepath.split(".")[0]
    out = cv2.VideoWriter(f"{savepath}inf_result/output_video2.mp4", fourcc, fps/step, (width, height))
    out2 = cv2.VideoWriter(f"{savepath}inf_result/depth.mp4", fourcc, fps/step, (width, height))
    
    # pdb.set_trace()
else:
    infe_range = [0,0]
    infe_camera = 99
    step = 1


for i in range(infe_range[0],infe_range[1]+1,step):
    # i/2の余りを計算
    remainder = i%2     
    device = torch.device(f"cuda:{remainder}" if torch.cuda.is_available() else "cpu")
    # pdb.set_trace()
    if data_type == "dataset":
        add_idx = i #range内のどこのデータを表示するか指定します。
        print(durations[i])       
        ## rgb, intrinsics, filepath, filename, depth_originを取得
        rgb = data['rgb'][(add_idx, infe_camera)]#(時間差,カメラ)
        intrinsics = get_from_dict(data, 'intrinsics')[(add_idx, infe_camera)]
        depth_origin = get_from_dict(data, 'depth')[(add_idx, infe_camera)]
        filepath = data["filename"][((add_idx, infe_camera))]#ファイルネームの取得（数字だけ）
        filename = filepath.split("/")[-1]
        filename = filename.split(".")[0]#数字部分のみ抽出して数値に変換
        # rgb, intrinsics, filepath, filename, depth_origin = DisplayDataset.infer(dataset,add_idx,infe_camera)#rgb:torch.Size([1, 3, 400, 879]),depth_origin:torch.Size([1, 1, 400, 879])
        
        # pdb.set_trace()
        rgb_disp = rgb.squeeze().cpu().detach().numpy()
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        rgb_disp = cv2.cvtColor((rgb_disp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)#(400, 879, 3)
        
    elif data_type == "VIDEO":
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()#frame:(720, 1280, 3)
        # フレームをクロップ
        frame = frame[y1:y2, x1:x2]
        intrinsics = torch.tensor(cam_para).unsqueeze(0)
        rgb = torch.tensor(frame).permute(2,0,1).unsqueeze(0)/255. #True:torch.Size([1, 3, 400, 879]),False:torch.Size([1, 3, 384, 640]),torch.float32
        depth_origin = rgb.mean(dim=1, keepdim=True)#torch.Size([1, 1, 384, 640])
        rgb_disp = frame
    else:
        intrinsics = torch.tensor(cam_para).unsqueeze(0)
        rgb = torch.tensor(cv2.imread(img_file)).permute(2,0,1).unsqueeze(0)/255. #True:torch.Size([1, 3, 400, 879]),False:torch.Size([1, 3, 384, 640]),torch.float32
        filepath = img_file
        filename = os.path.splitext(os.path.basename(filepath))
        savepath = filepath.split(".")[0]
        depth_origin = rgb.mean(dim=1, keepdim=True)#torch.Size([1, 1, 384, 640])
        rgb_disp = cv2.imread(img_file)


    ##時短用
    # rgbサイズを384*640に変換し、それに合わせてintrinsicsも変換する関数
    def resize_rgb_intrinsics(rgb, intrinsics):
        #圧縮する幅と高さを決める
        t_hight = 384
        t_width = 640
        #rgbの幅と高さを取得
        rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
        #rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
        resized_rgb = rgb.clone()
        # resized_rgb = F.interpolate(rgb, size=(384, 640), mode='bilinear', align_corners=False)
        # resized_rgb = F.interpolate(rgb, size=(t_hight, t_width), mode='bilinear', align_corners=False)
        resized_rgb = F.interpolate(rgb, size=(t_hight, t_width), mode='nearest')
        intrinsics2 = intrinsics.clone() 
        intrinsics2[0,0,0] = intrinsics[0,0,0].item()*t_width/rgb_w
        intrinsics2[0,0,2] = intrinsics[0,0,2].item()*t_width/rgb_w
        intrinsics2[0,1,1] = intrinsics[0,1,1].item()*t_hight/rgb_h
        intrinsics2[0,1,2] = intrinsics[0,1,2].item()*t_hight/rgb_h
        return resized_rgb, intrinsics2
       
    # def resize_to_nearest_32_multiple_and_max_pixels(rgb, intrinsics, max_pixels):
    #     # 元の画像の幅と高さを取得
    #     original_height, original_width = rgb.shape[2], rgb.shape[3]

    #     # オリジナルのアスペクト比を計算
    #     aspect_ratio = original_width / original_height

    #     # 最も近い32の倍数に調整
    #     new_height = round(original_height / 32) * 32
    #     new_width = round(new_height * aspect_ratio / 32) * 32

    #     # 指定された最大ピクセル数を超えないように調整
    #     while new_height * new_width > max_pixels:
    #         new_height -= 32
    #         new_width = round(new_height * aspect_ratio / 32) * 32

    #     # 画像をリサイズ
    #     # resized_rgb = F.interpolate(rgb, size=(new_height, new_width), mode='bilinear', align_corners=False)
    #     resized_rgb = F.interpolate(rgb, size=(new_height, new_width), mode='nearest')
    #     #nearest, bilinear, bicubic, area
    #     # 内部パラメータの調整
    #     intrinsics_scaled = intrinsics.clone()
    #     intrinsics_scaled[0, 0, 0] *= new_width / original_width
    #     intrinsics_scaled[0, 0, 2] *= new_width / original_width
    #     intrinsics_scaled[0, 1, 1] *= new_height / original_height
    #     intrinsics_scaled[0, 1, 2] *= new_height / original_height

    #     return resized_rgb, intrinsics_scaled

    

    rgb2, intrinsics2 = resize_rgb_intrinsics(rgb, intrinsics)
    # rgb2, intrinsics2 = resize_to_nearest_32_multiple_and_max_pixels(rgb, intrinsics, max_pixels=100000)
    # rgb2, intrinsics2 = resize_to_nearest_32_multiple_and_max_pixels(rgb, intrinsics, max_pixels=200000)
    # rgb2, intrinsics2 = resize_to_nearest_32_multiple(rgb, intrinsics)
    # pdb.set_trace()
    print("rgb.shape",rgb.shape)
    print("rgb2.shape",rgb2.shape)
    print("intrinsics",intrinsics)
    print("intrinsics2",intrinsics2)
    
    zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo="check")
    zerodepth_model.to(device)
    rgb2 = rgb2.to(device).detach()#True:torch.Size([1, 3, 384, 640]) False:
    intrinsics2 = intrinsics2.to(device).detach()#True:torch.Size([1, 3, 3])

    ## infer via ZeroDepth
    depth_pred = zerodepth_model(rgb2, intrinsics2)#True:torch.Size([1, 1, 384, 640]) False:torch.Size([1, 1, 384, 640])
    depth_pred = depth_pred.cpu().detach()#True:torch.Size([1, 1, 384, 640]) False:    
    depth_pred = torch.nn.functional.interpolate(depth_pred, size=(rgb.shape[2], rgb.shape[3]), mode='nearest')#predをオリジナルサイズに変更
    rgb2 = rgb2.cpu().detach()
    intrinsics2= intrinsics2.cpu().detach()
    rgb = rgb.cpu().detach()
    intrinsic = intrinsics2[0].cpu().detach()
    # print("depth_pred:",depth_pred.shape)#True:torch.Size([1, 1, 400, 879]) False:torch.Size([1, 1, 384, 640])
    # print("depth_origin:",depth_origin.shape)#True:torch.Size([1, 1, 400, 879]) False:torch.Size([1, 1, 384, 640])
    
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
    depth_pred_np = depth_pred.squeeze().cpu().detach().numpy()#True:(400, 879) False:(384, 640)
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
    # rgb_disp_gpu = torch.tensor(rgb_disp).to(YOLO_device)
    seg_results = seg_model.track(rgb_disp, show=False, conf=YOLO_conf, persist=True, save=False,
                                  save_txt=False, project=f"{savepath}inf_result/YOLO_seg{infe_camera}_{filename[0]}")#,
                                  #classes = [0,2,3,5,7,11])#person,car,motorcycle,bus,truck,9:traffic light,stop sign
    #retina_masks=True,
    seg_result = seg_results[0]

    pdb.set_trace()
    # print("seg_result:",seg_result_box.device)
    ## seg_result.masks.dataのshapeをtorch.Size([7, 320, 640])からtorch.Size([7, 400, 879])に変更
    if seg_result.masks is not None:
        seg_result_masks_gpu = seg_result.masks.data.to(device).detach() #torch.Size([3, 384, 640])
    
        # ターゲットのサイズ
        target_height = seg_result.orig_shape[0]
        target_width = seg_result.orig_shape[1]
        # ニアレストネイバー補間を使ってリサイズ
        resized_masks = F.interpolate(seg_result_masks_gpu.unsqueeze(0),  # バッチ次元を追加
                                    size=(target_height, target_width),   # 新しいサイズを指定
                                    mode='nearest').squeeze(0)            # バッチ次元を削除
        seg_result_masks_gpu = seg_result_masks_gpu.cpu()
        ## depth_predのセグメンテーション範囲のみを抽出
        # pdb.set_trace()
        # depth_pred = depth_pred.to(device)
        depth_map = depth_pred.to(device).squeeze()  # 結果はtorch.Size([400, 879])
        depth_true_map = depth_origin.to(device).squeeze() #torch.Size([400, 879])
        # Depthをマスクの数だけチャネル方向に複製します。
        depth_maps_expanded = depth_map.expand_as(resized_masks)#torch.Size([7,400, 879])
        depth_true_maps_expanded = depth_true_map.expand_as(resized_masks)#torch.Size([7,400, 879])
        depth_pred = depth_pred.cpu()
        depth_map = depth_map.cpu()    
        depth_origin = depth_origin.cpu()
        
        # マスクを適用して、マスクされた領域の予測Depth値と正解Depth値を取得
        masked_depth = depth_maps_expanded * resized_masks
        masked_true_depth = depth_true_maps_expanded * resized_masks
        depth_maps_expanded = depth_maps_expanded.cpu()
        depth_true_maps_expanded = depth_true_maps_expanded.cpu()
        resized_masks = resized_masks.cpu()
        masked_depth = masked_depth.cpu()
        masked_true_depth = masked_true_depth.cpu()
        # 0を除いた上で10-90%パーセンタイルの値を保持する関数
        def percentile_threshold_exclude_zero(tensor, lower_percentile, upper_percentile):
            # バッチの各要素に対してパーセンタイルを計算し適用
            percentile_masks = []
            for i in range(tensor.shape[0]):  # バッチサイズの次元をループ
                data = tensor[i][tensor[i] > 0].view(-1)  # 0を除外して1Dにフラット化
                if data.numel() == 0:  # データがない場合はマスクを適用しない
                    percentile_masks.append(torch.zeros_like(tensor[i], dtype=torch.bool))
                    continue
                k_lower = max(int(len(data) * lower_percentile / 100), 1)  # kは最小でも1である必要がある
                k_upper = min(int(len(data) * upper_percentile / 100), len(data))  # kは最大でdataの要素数
                k_upper = max(k_upper, k_lower)  # k_upperはk_lower以上であることを保証
                lower = torch.kthvalue(data, k_lower).values
                upper = torch.kthvalue(data, k_upper).values
                mask = (tensor[i] >= lower) & (tensor[i] <= upper)
                percentile_masks.append(mask)

            # マスクを適用して結果を返す
            return torch.stack(percentile_masks) * tensor
        # pdb.set_trace()
        # 0を除いたパーセンタイルフィルターを適用
        # pdb.set_trace()
        masked_depth = percentile_threshold_exclude_zero(masked_depth, 0, 100)
        masked_true_depth = percentile_threshold_exclude_zero(masked_true_depth, 0, 100)
      
        #マスク範囲を表示
        # mask_disp(masked_depth,masked_true_depth)
        
        ## 検出したモノの数だけループ
        ave_obj_depth = np.zeros(masked_depth.size(0))
        ave_obj_true_depth = np.zeros(masked_depth.size(0))
        posi_x = torch.zeros(masked_depth.size(0))
        posi_y = torch.zeros(masked_depth.size(0))
        posi = torch.zeros(masked_depth.size(0),2)

        #セグメンテーション画像上にDepthを追記
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
                # 0でない値だけを抽出します
                # obj_depth_o0 = obj_depth[obj_depth > 0]
                # obj_true_depth_o0 = obj_true_depth[obj_true_depth > 0]
                obj_depth_o0 = obj_depth[obj_depth > 0].cpu().detach().numpy()
                obj_true_depth_o0 = obj_true_depth[obj_true_depth > 0].cpu().detach().numpy()
                # 平均値を計算します。
                # ave_obj_depth[j] = torch.mean(obj_depth_o0)
                # ave_obj_true_depth[j] = torch.mean(obj_true_depth_o0)
                ave_obj_depth[j] = np.mean(obj_depth_o0)
                ave_obj_true_depth[j] = np.mean(obj_true_depth_o0)
            else:
                pass  # 何もしない
        
            #位置計算
            posi_x[j] = (seg_result.boxes.xyxy[j][0]+seg_result.boxes.xyxy[j][2])/2
            posi_y[j] = (seg_result.boxes.xyxy[j][1]+seg_result.boxes.xyxy[j][3])/2
            posi[j] = torch.tensor([posi_x[j],posi_y[j]])
            
            
            ## Depthを画像上の位置に表示
            # テキスト設定
            ave_obj_depth_text = np.round(ave_obj_depth[j],1)
            ave_obj_true_depth_text = np.round(ave_obj_true_depth[j],1)
            # ave_obj_depth_text = round(ave_obj_depth[j].item(),1)
            # ave_obj_true_depth_text = round(ave_obj_true_depth[j].item(),1)
            pred_text = f"{ave_obj_depth_text}"  # 画像に挿入するテキスト
            true_text = f"{ave_obj_true_depth_text}"  # 画像に挿入するテキスト
            org = (int(posi_x[j]),int(posi_y[j]))  # テキストを挿入する位置（左下の座標）
            font = cv2.FONT_HERSHEY_SIMPLEX  # フォントタイプ
            fontScale = 1  # フォントサイズ
            color_b = (255, 0, 0)  # テキストの色（BGR）
            color_r = (0, 0, 255)  # テキストの色（BGR）
            thickness = 2  # テキストの太さ
            
            # テキストを画像に挿入
            cv2.putText(rgb_text, pred_text, org, font, fontScale, color_b, thickness, cv2.LINE_AA)
            cv2.putText(rgb_true_text, true_text, org, font, fontScale, color_r, thickness, cv2.LINE_AA)    
    else:
        rgb_text = seg_result.plot().copy()#rgb_disp.copy()
        rgb_true_text = rgb_disp.copy()
        ave_obj_true_depth = np.array([0])
        ave_obj_depth = np.array([0])
        print("物体検出なし")
    
    
    # データを集約
    ave_obj_true_depth_all = np.concatenate([ave_obj_true_depth_all,ave_obj_true_depth])
    ave_obj_depth_all = np.concatenate([ave_obj_depth_all,ave_obj_depth])
    # ave_obj_true_depth_all = np.concatenate([ave_obj_true_depth_all,ave_obj_true_depth.detach().numpy()])
    # ave_obj_depth_all = np.concatenate([ave_obj_depth_all,ave_obj_depth.detach().numpy()])
    # print("rgb_text:",rgb_text.shape)#(720, 1280, 3)
    ## フォルダを作成
    if not os.path.exists(f"{savepath}inf_result"):
        os.mkdir(f"{savepath}inf_result")
    os.chmod(f"{savepath}inf_result", 0o777)
    if not os.path.exists(f"{savepath}inf_result/hoge"):
        os.mkdir(f"{savepath}inf_result/hoge")
    os.chmod(f"{savepath}inf_result", 0o777)

    ## input（RGB）を保存
    # cv2.imwrite(f"{savepath}inf_result/rgb_input{infe_camera}_{filename[0]}.png", rgb_disp)
    cv2.imwrite(f"{savepath}inf_result/hoge/rgb_text{infe_camera}_{filename}_{i}.png", rgb_text)
    # cv2.imwrite(f"{savepath}inf_result/rgb_true_text{infe_camera}_{filename[0]}.png", rgb_true_text)
    # depthのoutput（カラー、TRIバージョン）を保存
    # write_image(f"{savepath}inf_result/depth_c_inv{infe_camera}_{filename}_{i}.png", depth_rgb_image_vizdepth)
    # write_image(f"{savepath}inf_result/depth_true_inv{infe_camera}_{filename}.png", depth_origin_image_vizdepth)
    # write_image(f"{savepath}inf_result/loss_{infe_camera}_{filename[0]}.png", loss_image_vizdepth)
    # write_image(f"{savepath}inf_result/loss_abs{infe_camera}_{filename}.png", loss_abs_image_vizdepth)
    # write_image(f"{savepath}inf_result/loss_abs_o2_{infe_camera}_{filename}.png", loss_abs_o10_image_vizdepth)#Depthの誤差がXよりも大きいところだけ保存
    ##動画として保存
    # pdb.set_trace()
    if data_type == "VIDEO" or "dataset":
        out.write(rgb_text)
        depth_rgb_image_vizdepth2= (255 * depth_rgb_image_vizdepth[:, :, ::-1]).astype(np.uint8)
        # image = cv2.imread(depth_rgb_image_vizdepth)
        # # 読み込んだ画像データがnumpy.ndarrayオブジェクトなのを確認して、
        # # video writerに書き込みます。
        # if image is not None and isinstance(image, np.ndarray):
        #     out2.write(depth_rgb_image_vizdepth2)
        # else:
        #     print("読み込んだ画像が正しくありません")
        out2.write(depth_rgb_image_vizdepth2)
    else:
        pass
    # print(f"{i}フレーム目、読み込み失敗")
        
    #データをint型で1次元に変換
    depth_pred_np = depth_pred_np.astype(np.int32).flatten()
    depth_origin_np = depth_origin_np.astype(np.int32).flatten()
    loss_np = loss_np.astype(np.int32).flatten()
    loss_abs_np = loss_abs_np.astype(np.int32).flatten()
    
    #結果をヒストグラムでプロット
    if seg_result.masks is not None:   
        compare_distribution(masked_true_depth, masked_depth, savepath, infe_camera, filename)#各物体の距離分布
        # compare_distribution(filtered_masked_true_depth_ex_zero, filtered_masked_depth_ex_zero, savepath, infe_camera, filename)#各物体の距離分布（パーセンタイル補正後）
        # compare_true_pred(ave_obj_true_depth, ave_obj_depth, savepath, infe_camera, filename)#各物体の距離プロット比較
        # plot_distribution(depth_origin_np, depth_pred_np, loss_np, loss_abs_np, savepath, infe_camera, filename)
    compare_true_pred(ave_obj_true_depth_all, ave_obj_depth_all, savepath, infe_camera, filename="all")#各物体の距離プロット比較（データ全体）
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    del zerodepth_model,depth_pred,rgb,rgb2,intrinsics,intrinsics2,depth_pred_np,depth_origin_np,loss_np,loss_abs_np
    del seg_results,depth_origin
    if seg_result.masks is not None:
        del seg_result_masks_gpu
        del depth_map, depth_maps_expanded, masked_depth, masked_true_depth, posi_x, posi_y, posi, rgb_text,resized_masks, depth_true_maps_expanded

    torch.cuda.empty_cache()
    print(f"{i}ループ目終了")
    print(f"データ{filename}を{savepath}inf_resultに保存したよ")

subprocess.run(['chmod', '-R', '777', f'{savepath}inf_result'], check=True)    
    
if data_type == "VIDEO":
    out.release()
    out2.release()        
    cap.release()
print("最後まできたよ")

