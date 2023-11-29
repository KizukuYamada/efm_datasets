import numpy as np
import torch

# パーセンタイルに基づく外れ値の除去
def remove_outliers(data, lower_percentile, upper_percentile):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

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