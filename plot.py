import numpy as np
import matplotlib
# matplotlib.use('fast')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
from efm_datasets.utils.config import read_config
from efm_datasets.utils.setup import setup_dataset
from efm_datasets.utils.data import make_batch, fold_batch, get_from_dict, interleave_dict, modrem
from efm_datasets.utils.viz import viz_depth
from efm_datasets.utils.write import write_image
from scripts.display.DisplayDataset import DisplayDataset

#
loaded_data = np.load('temp_file.npz')
# print(loaded_data.files)
depth_pred_np = loaded_data['arr_0']
depth_origin_np = loaded_data['arr_1']
loss_np = loaded_data['arr_2']
loss_abs_np = loaded_data['arr_3']
savepath = loaded_data['arr_4']
infe_camera = loaded_data['arr_5']
filename = loaded_data['arr_6']

#savepathの文字列を取得
print(savepath)# = savepath[0]

print("depth_pred_np:",depth_pred_np.shape)
print("depth_pred_np:",depth_pred_np)

#データをint型で1次元に変換
depth_pred_np = depth_pred_np.astype(np.int32).flatten()
depth_origin_np = depth_origin_np.astype(np.int32).flatten()
loss_np = loss_np.astype(np.int32).flatten()
loss_abs_np = loss_abs_np.astype(np.int32).flatten()
print("depth_pred_np:",depth_pred_np)

# 結果をヒストグラムでプロット

fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列のsubplotを作成

# 1つ目のグラフ
labels = [f'True (Max:{np.max(depth_origin_np)})', f'Pred (Max:{np.max(depth_pred_np)})']
axes[0].hist([depth_origin_np[depth_origin_np>=0],depth_pred_np[depth_pred_np>=0]], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels)#, density=True
# axes[1].hist(depth_origin_np[depth_origin_np>=0], range=(0, 99), bins=10, alpha=0.7,bottom=0,label=labels)
axes[0].set_title('Depth Data Distribution')
axes[0].set_xlabel('Distance[m]')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# # 1つ目のグラフ(rate表示用)
# labels = [f'True (Max:{np.max(depth_origin_np)})', f'Pred (Max:{np.max(depth_pred_np)})']
# axes[0].hist(depth_origin_np[depth_origin_np>=0], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels, density=True)#, density=True
# axes[0].hist(depth_pred_np[depth_pred_np>=0], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels[1], density=True)#, density=True
# axes[0].set_title('Depth Data Distribution')
# axes[0].set_xlabel('Distance[m]')
# axes[0].set_ylabel('Frequency')
# axes[0].legend()

# 2つ目のグラフ
label2 = [f'Mean:{round(np.mean(loss_np),2)}\nMax:{np.max(loss_np)}']
axes[1].hist(loss_abs_np, range=(0, 99), bins=20, alpha=0.7,bottom=0,label=label2)
axes[1].set_title('Abs_Loss Distribution (Metric Scale)')
axes[1].set_xlabel('Loss Value[m]')
axes[1].set_ylabel('Frequency')
axes[1].legend()



plt.tight_layout()
plt.savefig(f"loss_abs_plot{infe_camera}_{filename}.png")
plt.show()

#特定の値表示用
mask_o10 = (loss_abs>=10).float()
loss_abs_o10 = loss_abs * mask_o10 
# viz_depthを使ってdepthを可視化
depth_rgb_image_vizdepth = viz_depth(depth_pred, filter_zeros=True) 
depth_origin_image_vizdepth = viz_depth(depth_origin, filter_zeros=True) 
loss_image_vizdepth = viz_depth(loss, filter_zeros=True)
loss_abs_image_vizdepth = viz_depth(loss_abs, filter_zeros=True)
loss_abs_o10_image_vizdepth = viz_depth(loss_abs_o10, filter_zeros=True)