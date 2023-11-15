import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    
import pdb
import math
    
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
    plt.show()  
    
def compare_distribution(masked_true_depth, masked_depth, savepath, infe_camera, filename):
    #データをnpのint型で1次元に変換
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
    plt.savefig(f"{savepath}inf_result/obj_depth_{infe_camera}_{filename}.png")
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