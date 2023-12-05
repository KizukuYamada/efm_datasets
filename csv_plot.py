import pandas as pd
import matplotlib.pyplot as plt
import os

# # CSVファイルの読み込み
# # 仮のファイルパスを使用します。実際のファイルパスに置き換えてください。
# file_path = '/mnt/fsx/datasets/Platooning/20231030/20231030/nobori/@data_20231030163856.csv'

# # CSVデータを読み込む
# data = pd.read_csv(file_path)

# # 特定の時刻範囲のデータを選択
# # 例として、時刻 0.1 から 0.5 までのデータを抽出します。
# # 時刻の列名を 'time_Rate1[sec]' など正確な名前に置き換えてください。
# start_time = 0.1
# end_time = 100
# time_column = '#time_Rate1[sec]'
# subset_data = data[(data[time_column] >= start_time) & (data[time_column] <= end_time)]

# # 時系列グラフの描画
# plt.figure(figsize=(15, 5))

# # 時刻をx軸にして、他の列をy軸にプロット
# # ここでは、'ltc_afl_g_fsn_dx_last[0]' という列を例としてプロットしています。
# # 必要な列すべてをプロットするには、列名をループで回してプロットする必要があります。
# # ここでは、'ltc_afl_g_fsn_dx_last' で始まる列をプロットする例を示します。

# for column in data.columns:
#     if column.startswith('ltc_afl_g_fsn_dx_last'):
#         plt.plot(subset_data[time_column], subset_data[column], label=column)

# plt.xlabel('Time (sec)')
# plt.ylabel('Values')
# plt.title('Time Series Plot from CSV Data')
# plt.legend()
# plt.show()

def plot_time_series_from_csv(file_path, start_time, end_time, savefinalpath):
    """
    Reads a CSV file and plots selected time series data within a specified time range as subplots.
    Automatically determines the number of data columns and their names.

    :param file_path: The path to the CSV file.
    :param start_time: The start of the time range for data selection.
    :param end_time: The end of the time range for data selection.
    """
    # CSVデータを読み込む（ヘッダーをスキップして）
    # 最初の数行はヘッダー情報なのでスキップし、列名を自動的に取得
    data = pd.read_csv(file_path, skiprows=0)

    # 特定の時刻範囲のデータを選択
    time_column = data.columns[0]  # 最初の列を時間と仮定
    subset_data = data[(data[time_column] >= start_time) & (data[time_column] <= end_time)]

    # 信号ごとにサブプロットを作成
    signal_columns = data.columns[1:]  # 最初の列以外を信号データの列として取得
    num_signals = len(signal_columns)
    
    # サブプロットの作成
    fig, axs = plt.subplots(num_signals, 1, figsize=(10, num_signals * 2), sharex=True)
    
    if num_signals == 1:  # もし信号が1つだけある場合の処理
        axs = [axs]  # axsをリストに変換してイテレーション可能にする
    
    for ax, column in zip(axs, signal_columns):
        ax.plot(subset_data[time_column], subset_data[column], label=column)
        # ax.set_title(column)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{savefinalpath}/true_time_flow.png")
    plt.show()
    
if __name__ == '__main__':
    # 仮のファイルパスを使用
    savepath = "/mnt/fsx/datasets/Platooning/20231030/20231030/nobori/20231030_065630000_iOS"

    file_path = f'{os.path.dirname(savepath)}/@data_20231030163856.csv'  # このパスを実際のものに変更してください
    savefinalpath = '/mnt/fsx/datasets/Platooning/20231030/20231030/nobori/'
    # 関数を呼び出してプロット
    plot_time_series_from_csv(file_path, 0, 1927.2,savefinalpath)