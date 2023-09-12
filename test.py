def print_hello():
    print("hello")
    return 0

# コメントアウト
# 距離を推定する関数


# この関数はCSVファイルを読み込んで、リストのリストを返します
def read_csv(filename):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        # リストのリストを作成します
        data = list(csvreader)
    return data


