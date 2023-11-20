import pdb

def count_nines_and_calculate_percentage(file_path):
    # 各列の9の数と全要素数をカウントするための辞書
    count_nines = {2: 0, 3: 0, 4: 0}
    total_elements = {2: 0, 3: 0, 4: 0}

    # ファイルを開いて各行を読み込む
    with open(file_path, 'r') as file:
        for line in file:
            # 各行を空白で分割して数値に変換
            numbers = [int(num) for num in line.split()]

            # 各列における9の数と全要素数をカウント
            
            for i in range(1, 4):
                # pdb.set_trace()
                total_elements[i+1] += 1
                if numbers[i] == 9:
                    count_nines[i+1] += 1

    print(count_nines)
    print(total_elements)
    # 各列における9の割合を計算
    percentage_nines = {}
    for i in range(2, 5):
        if total_elements[i] > 0:
            percentage_nines[i] = (count_nines[i] / total_elements[i]) * 100
        else:
            percentage_nines[i] = 0

    return count_nines, total_elements, percentage_nines

# 使用例
file_path = '/mnt/fsx/datasets/Platooning/GTP4_vision_preview2/answerGPT_30000_1000_79000.txt'  # ファイルパスを指定
nines_count, total_elements, nines_percentage = count_nines_and_calculate_percentage(file_path)
print(f"白い車レーンが不明の数: {nines_count[2]} / {total_elements[2]} , 割合: {nines_percentage[2]:.2f}%")
print(f"自車レーンが不明の数: {nines_count[3]} / {total_elements[3]} , 割合: {nines_percentage[3]:.2f}%")
print(f"工事区間か不明の数: {nines_count[4]} / {total_elements[4]} , 割合: {nines_percentage[4]:.2f}%")