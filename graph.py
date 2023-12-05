import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    
import pdb
import math
import cv2 
from efm_datasets.utils.viz import viz_depth
import torch
import os
import csv_plot
import pandas as pd
    
def plot_distribution(depth_origin_np, depth_pred_np, loss_np,loss_abs_np, savepath, infe_camera, filename):

    #結果をヒストグラムでプロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列のsubplotを作成
    
    
    

    # 1つ目のグラフ
    # labels = [f'True (Max:{np.max(depth_origin_np)})', f'Pred (Max:{np.max(depth_pred_np)})']
    # axes[0].hist([depth_origin_np[depth_origin_np>=0],depth_pred_np[depth_pred_np>=0]], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels)#, density=True
    # # axes[1].hist(depth_origin_np[depth_origin_np>=0], range=(0, 99), bins=10, alpha=0.7,bottom=0,label=labels)
    # axes[0].set_title('Depth Data Distribution')
    # axes[0].set_xlabel('Distance[m]')
    # axes[0].set_ylabel('Frequency')
    # axes[0].legend()

    # # 1つ目のグラフ(rate表示用)
    labels = [f'True (Max:{np.max(depth_origin_np)})', f'Pred (Max:{np.max(depth_pred_np)})']
    axes[0].hist(depth_origin_np[depth_origin_np>=0], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels, density=True)#, density=True
    axes[0].hist(depth_pred_np[depth_pred_np>=0], range=(0, 119), bins=12, alpha=0.7, bottom=0,label=labels[1], density=True)#, density=True
    axes[0].set_title('Depth Data Distribution')
    axes[0].set_xlabel('Distance[m]')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # 2つ目のグラフ
    label2 = [f'Mean:{round(np.mean(loss_np),2)}\nMax:{np.max(loss_np)}']
    axes[1].hist(loss_abs_np, range=(10, 99), bins=90, alpha=0.7,bottom=0,label=label2)
    axes[1].set_title('Abs_Loss Distribution (Metric Scale)')
    axes[1].set_xlabel('Loss Value[m]')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{savepath}inf_result/loss_abs_plot2{infe_camera}_{filename}.png")
    # plt.show()  
    
def compare_distribution(masked_true_depth, masked_depth, savefinalpath, infe_camera, filename):
    #データをnpのint型で1次元に変換
    if torch.is_tensor(masked_true_depth):
        masked_true_depth_np = masked_true_depth.cpu().detach().numpy()
        masked_pred_depth_np = masked_depth.cpu().detach().numpy()
    
    #9:16表示に最適な最適なサブプロット行列を見つける
    # pdb.set_trace()
    if masked_pred_depth_np.shape[0] == 1:
        fig, ax = plt.subplots()
        k=0
        obj_k_true = masked_true_depth_np[k][masked_true_depth_np[k] > 0].flatten()
        obj_k_pred = masked_pred_depth_np[k][masked_pred_depth_np[k] > 0].flatten()
        if len(obj_k_true) == 0:
            obj_k_true = np.zeros((2, 2), dtype=np.float32).flatten()
        if len(obj_k_pred) == 0:
            obj_k_pred = np.zeros((2, 2), dtype=np.float32).flatten()
        dis_min = int(min(np.min(obj_k_true),np.min(obj_k_pred)))-5
        dis_max = int(max(np.max(obj_k_true),np.max(obj_k_pred)))+5
        labels = [f'LiDAR ({int(np.min(obj_k_true))}-{int(np.max(obj_k_true))})', f'Pred ({int(np.min(obj_k_pred))}-{int(np.max(obj_k_pred))})']
        ax.hist([obj_k_true,obj_k_pred], range=(dis_min, dis_max), bins=dis_max-dis_min+1, alpha=0.7, bottom=0,label=labels,density=True)#, density=True
        
        ax.axvline(np.mean(obj_k_true), color='blue', linestyle='dashed', linewidth=2, label=f'LiDAR Mean: {np.mean(obj_k_true):.1f}')
        ax.axvline(np.mean(obj_k_pred), color='red', linestyle='dashed', linewidth=2, label=f'Pred Mean: {np.mean(obj_k_pred):.1f}')
        ax.legend()
        ax.set_title(f'Object {k+1} ')
        ax.set_xlabel('Distance[m]')
        ax.set_ylabel('Rate[]')
    else:
        nrow, ncol = find_closest_ratio(masked_pred_depth_np.shape[0])
        takasa = 3*nrow
        
        # hoge = round((masked_pred_depth_np.shape[0]+1)/2)
        piyo = 3*ncol
        
        # hoge = round(masked_pred_depth_np.shape[0].astype(np.int32)/2)
        # piyo = round(6*round(masked_pred_depth_np.shape[0].astype(np.int32)/2)/2)
        #結果をヒストグラムでプロット

        
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(piyo, takasa))  # 1行2列のsubplotを作成
        # pdb.set_trace()
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=0)  # axesを2次元配列に変換します。
        
        for k in range(masked_depth.size(0)):
            obj_k_true = masked_true_depth_np[k][masked_true_depth_np[k] > 0].flatten()
            obj_k_pred = masked_pred_depth_np[k][masked_pred_depth_np[k] > 0].flatten()
            if len(obj_k_true) == 0:
                obj_k_true = np.zeros((2, 2), dtype=np.float32).flatten()
            if len(obj_k_pred) == 0:
                obj_k_pred = np.zeros((2, 2), dtype=np.float32).flatten()
            dis_min = int(min(np.min(obj_k_true),np.min(obj_k_pred)))-5
            dis_max = int(max(np.max(obj_k_true),np.max(obj_k_pred)))+5
            # print("k",k)
            # print("row",k // ncol)
            # print("col",k % ncol)
            # pdb.set_trace()
            row = k // ncol
            col = k % ncol
            # # 1つ目のグラフ(rate表示用)
            # pdb.set_trace()
            labels = [f'LiDAR ({int(np.min(obj_k_true))}-{int(np.max(obj_k_true))})', f'Pred ({int(np.min(obj_k_pred))}-{int(np.max(obj_k_pred))})']
            axes[row,col].hist([obj_k_true,obj_k_pred], range=(dis_min, dis_max), bins=dis_max-dis_min+1, alpha=0.7, bottom=0,label=labels,density=True)#, density=True
            
            axes[row,col].axvline(np.mean(obj_k_true), color='blue', linestyle='dashed', linewidth=2, label=f'LiDAR Mean: {np.mean(obj_k_true):.1f}')
            axes[row,col].axvline(np.mean(obj_k_pred), color='red', linestyle='dashed', linewidth=2, label=f'Pred Mean: {np.mean(obj_k_pred):.1f}')
            axes[row,col].legend()
            axes[row,col].set_title(f'Object {k+1} ')
            axes[row,col].set_xlabel('Distance[m]')
            axes[row,col].set_ylabel('Rate[]')
        


    plt.tight_layout()
    plt.savefig(f"{savefinalpath}/obj_depth_{infe_camera}_{filename}.png")
    # plt.show()  
    
    
def find_closest_ratio(a):
    best_diff = float('inf')
    best_x, best_y = 1, a  # 初期値
    if a <= 4:
        range_num = 0
    else:
        range_num = 2
    
    for hoge in range(0, range_num):
        for x in range(1, a):
            y = math.ceil(a / x)
            if x * y <= a + hoge:
                continue

            # 現在の比率と9:16の比率の差を計算
            current_diff = abs((x / y) - (9 / 16))

            if current_diff < best_diff:
                best_diff = current_diff
                best_x, best_y = x, y

    return best_x, best_y

def mask_disp(masked_depth,masked_true_depth):
    for l in range(masked_depth.size(0)):
            masked_depth_np = viz_depth(masked_depth[l], filter_zeros=True) 
            masked_true_depth_np = viz_depth(masked_true_depth[l], filter_zeros=True) 
            cv2.imshow(f'Depth_True Map{l}', masked_true_depth_np)
            cv2.imshow(f'Depth Map{l}', masked_depth_np)
            # key = cv2.waitKey(0)
    return
            
            
def plot_time():
    return
            
def compare_true_pred(ave_obj_true_depth, ave_obj_depth, savefinalpath, infe_camera, filename):
    axes = plt.subplots(1, 3, figsize=(12, 4))
    # GPU上にあるテンソルの場合、まずCPUに移動させる
    if torch.is_tensor(ave_obj_true_depth):
        ave_obj_true_depth = ave_obj_true_depth.cpu().detach().numpy()
        # if ave_obj_depth.is_cuda:
        ave_obj_depth = ave_obj_depth.cpu().detach().numpy()
    #桁数を丸める
    ave_obj_true_depth_np = np.round(ave_obj_true_depth,1)
    ave_obj_depth_np = np.round(ave_obj_depth,1)
    #距離の小さい順にソート？必要？
    # データサイズ分の配列を作成
    x = np.arange(ave_obj_true_depth_np.shape[0])
    # 誤差[m]を計算
    perd_loss = ave_obj_true_depth_np - ave_obj_depth_np
    # 平均絶対誤差[m]を計算
    average_perd_loss = np.mean(abs(perd_loss))
    # 絶対差分[%]を計算
    diff_percent = np.abs(perd_loss / ave_obj_true_depth_np * 100)
    # 平均絶対差分[%]を計算
    average_diff_percent = np.mean(diff_percent)
    # pdb.set_trace()c
    labels = ['True', 'Pred']
    plt.subplot(1, 3, 1)
    plt.scatter(x,ave_obj_true_depth_np,label="True")
    plt.scatter(x,ave_obj_depth_np,label="Pred")
    # plt.xticks(range(min(x), max(x)+1, 1))
    plt.title('Object Distance')
    plt.xlabel('Object[]')
    plt.ylabel('Distance[m]')
    plt.legend()

    # 2つ目のグラフ
    plt.subplot(1, 3, 2)
    plt.scatter(ave_obj_true_depth_np,perd_loss,c="purple")
    plt.xlim(0, int(max(ave_obj_true_depth_np))+5)
    plt.title('True-Pred Loss')
    plt.xlabel('True Distance[m]')
    plt.ylabel('True-Pred Loss[m]')
    plt.axhline(y=average_perd_loss, color='r', linestyle='--', label=f'Average Loss:{average_perd_loss:.1f}[m]')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.scatter(ave_obj_true_depth_np,diff_percent,c="purple")
    plt.xlim(0, int(max(ave_obj_true_depth_np))+5)
    plt.title('(True-Pred)/True Loss[%]')
    plt.xlabel('True Distance[m]')
    plt.ylabel('(True-Pred)/True Loss[%]')
    # plt.legend()
    # 平均差分[%]の線を追加
    plt.axhline(y=average_diff_percent, color='r', linestyle='--', label=f'Average Loss:{average_diff_percent:.1f}[%]')
    plt.legend()
    # plt.scatter(x,ave_obj_true_depth_np, c="yellow", marker="*", alpha=0.5,
    #         linewidths="2", edgecolors="orange")
    # plt.scatter(x,ave_obj_depth_np, c="blue", marker="*", alpha=0.5,
    #         linewidths="2", edgecolors="orange")
    # plt.title('Depth Data Distribution')
    # plt.xlabel('Distance[m]')
    # plt.ylabel('Frequency')
    # plt.legend()

    # グラフを表示
    plt.tight_layout()  # グラフ間のスペースを調整
    plt.savefig(f"{savefinalpath}/Compare_TP_{infe_camera}_{filename}.png")
    # plt.show()
    
    return

import matplotlib.pyplot as plt
import numpy as np

def plot_combined_time_series(time, depth, bb_x, bb_y, tracking,savefinalpath,data_type,flame_num, fps):
    """
    Plots 'depth', 'bb_x', 'bb_y' time series data in the same subplot with different legends,
    and 'tracking' in a separate subplot. All plots share the 'time' x-axis.

    :param time: Time series data for the x-axis.
    :param depth: Time series data for 'depth' with multiple columns.
    :param bb_x: Time series data for 'bb_x' with multiple columns.
    :param bb_y: Time series data for 'bb_y' with multiple columns.
    :param tracking: Time series data for 'tracking' (single column).
    """
    # Create a figure and two subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Plot 'depth', 'bb_x', 'bb_y' in the first subplot
    for col in range(depth.shape[0]):
        ax1.plot(time, depth[col,:], label=f'ID {col+1}')
    if data_type == "VIDEO":
        true_file_path = f'{os.path.dirname(savepath)}/@data_20231030163856 (copy).csv'
        dif_time_fmcsv = 21
        disp_time_csv = [flame_num[0]/fps-dif_time_fmcsv,flame_num[1]/fps-dif_time_fmcsv]
        
        # pdb.set_trace()
        data = pd.read_csv(true_file_path, skiprows=4)
        specific_signal_name = 'ltc_afl_g_fsn_dx_sel[0]'
        specific_signal_name1 = 'ltc_afl_g_fsn_dx_sel[1]'
        specific_signal_name2 = 'ltc_afl_g_fsn_dx_sel[2]'
        specific_signal_name3 = 'ltc_afl_g_fsn_dx_sel[3]'
        specific_signal_name4 = 'ltc_afl_g_fsn_dx_sel[4]'
        specific_signal_name5 = 'ltc_afl_g_fsn_dx_sel[5]'
        # last0 = 'ltc_afl_g_fsn_dx_last[0]'
        # last1 = 'ltc_afl_g_fsn_dx_last[1]'
        # last2 = 'ltc_afl_g_fsn_dx_last[2]'
        # last3 = 'ltc_afl_g_fsn_dx_last[3]'
        # last4 = 'ltc_afl_g_fsn_dx_last[4]'
        # last5 = 'ltc_afl_g_fsn_dx_last[5]'
        # last6 = 'ltc_afl_g_fsn_dx_last[6]'
        # last7 = 'ltc_afl_g_fsn_dx_last[7]'
        # last8 = 'ltc_afl_g_fsn_dx_last[8]'
        # last9 = 'ltc_afl_g_fsn_dx_last[9]'
        time_column = data.columns[0]
        
        sel_data = data[(data[time_column] >= disp_time_csv[0]) & (data[time_column] <= disp_time_csv[1])]
        signal_columns = [col for col in data.columns if specific_signal_name in col]
        n0_distance_data = data[(data[signal_columns] >= disp_time_csv[0]) & (data[signal_columns] <= disp_time_csv[1])]
        # pdb.set_trace()
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name], linestyle=':', label=f'{specific_signal_name}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name1], linestyle=':', label=f'{specific_signal_name1}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name2], linestyle=':', label=f'{specific_signal_name2}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name3], linestyle=':', label=f'{specific_signal_name3}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name4], linestyle=':', label=f'{specific_signal_name4}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name5], linestyle=':', label=f'{specific_signal_name5}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name], linestyle=':', label=f'{specific_signal_name}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name1], linestyle=':', label=f'{specific_signal_name1}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name2], linestyle=':', label=f'{specific_signal_name2}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name3], linestyle=':', label=f'{specific_signal_name3}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name4], linestyle=':', label=f'{specific_signal_name4}')
        ax1.plot(sel_data["#time_Rate1[sec]"]-sel_data["#time_Rate1[sec]"].iloc[0], sel_data[specific_signal_name5], linestyle=':', label=f'{specific_signal_name5}')
    y_min = -20  # y軸の最小値
    y_max = 80  # y軸の最大値
    ax1.set_ylim(y_min, y_max)
    ax1.set_title('Depth')
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2, fontsize='small')
    
        # Plot 'depth', 'bb_x', 'bb_y' in the first subplot
    for col in range(depth.shape[0]):
        ax2.plot(time, bb_x[col,:], linestyle='--')
        
    ax2.set_title('BB_X')
    ax2.legend()

    for col in range(depth.shape[0]):
        ax3.plot(time, bb_y[col,:], linestyle=':')

    ax3.set_title('BB_Y')
    ax3.legend()

    # Plot 'tracking' in the second subplot
    ax4.plot(time, tracking, label='Tracking', color='k')
    ax4.set_title('Tracking')
    ax4.legend()

    # Set a common x-label
    plt.xlabel('Time')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"{savefinalpath}/time_flow.png")
    plt.show()

# Example usage with sample data
# Generating sample data
# np.random.seed(0) # For reproducibility
# time_points = 100 # Number of time points

# # Create sample data for each time series
# time_data = np.linspace(0, 10, time_points)
# depth_data = np.random.randn(time_points).cumsum()
# bb_x_data = np.random.randn(time_points).cumsum()
# bb_y_data = np.random.randn(time_points).cumsum()
# tracking_data = np.random.randn(time_points).cumsum()

# #時短用
if __name__ == '__main__':
    savepath = "/data/datasets/Platooning/20231030/20231030/nobori/20231030_065630000_iOS"
    savefinalpath = "/data/datasets/Platooning/20231030/20231030/nobori/20231030_065630000_iOSinf_result/x300500y250400_f3000031000_s1"
    loaded_data = np.load(f"{savefinalpath}/data.npz")
    time_data = loaded_data['arr_0']
    tracking_data = loaded_data['arr_1']
    depth_data = loaded_data['arr_2']
    bb_x_data = loaded_data['arr_3']
    bb_y_data = loaded_data['arr_4']
    print("time_data",time_data.shape)
    print("depth_data",depth_data.shape)
    print("bb_x_data",bb_x_data.shape)
    print("bb_y_data",bb_y_data.shape)
    print("tracking_data",tracking_data.shape)
    data_type = "VIDEO"
    flame_num = [30000,31000]
    fps = 30
    # pdb.set_trace()

    # Plot the time series
    plot_combined_time_series(time_data, depth_data, bb_x_data, bb_y_data, tracking_data, savefinalpath,data_type,flame_num, fps)

