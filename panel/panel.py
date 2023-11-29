import pandas as pd
import matplotlib.pyplot as plt
import pdb

# CSVファイルの読み込み
# ここでは 'date' 列を日付として解釈し、インデックスとして設定します
file_path = '/mnt/fsx/datasets/Platooning/20231030/20231030/kudari/ninshiki_all.xlsm'
df = pd.read_excel(file_path, sheet_name='deta')#, index_col='date', parse_dates=True)

# データのプロット
# 'your_column_name' はプロットしたいデータ列の名前に置き換えてください
pdb.set_trace()
df['ltc_afl_g_fsn_dx_last[0]'].plot()

# グラフのタイトルとラベルの設定
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')

# グラフの表示
plt.show()