import torch.nn.functional as F
import numpy as np
import torch

# rgbサイズを384*640に変換し、それに合わせてintrinsicsも変換する関数
def resize_rgb_intrinsics(rgb, intrinsics):
    #圧縮する幅と高さを決める
    t_hight = 384
    t_width = 640
    #rgbの幅と高さを取得
    rgb_h, rgb_w = rgb.shape[2], rgb.shape[3]
    #rgbを[1,3,height,width]の形状から[1,3,384,640]の形状に変換
    resized_rgb = rgb.clone()
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

def adjust_intrinsics(cam_para, x, y, width, height):
    # リストをテンソルに変換
    cam_para_tensor = torch.tensor(cam_para, dtype=torch.float32)

    # 必要な変更を加える
    # 例えば、焦点距離のスケーリングや中心座標の変更など
    cam_para_tensor[0, 2] -= x  # Adjust c_x
    cam_para_tensor[1, 2] -= y  # Adjust c_y

    # 他の必要な変更をここに追加

    # テンソルをリストに変換
    adjusted_cam_para = cam_para_tensor.tolist()

    return adjusted_cam_para

# 例
# intrinsics = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
# crop_coords = (x, y, w, h)
# new_intrinsics = adjust_intrinsics(intrinsics, crop_coords)