import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    
    
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