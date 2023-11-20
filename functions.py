import numpy as np

# パーセンタイルに基づく外れ値の除去
def remove_outliers(data, lower_percentile, upper_percentile):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]
