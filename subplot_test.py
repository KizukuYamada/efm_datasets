import math



def find_closest_ratio(a):
    best_diff = float('inf')
    best_x, best_y = 1, a  # 初期値

    for hoge in range(0, 2):
        for x in range(1, a + 1):
            y = math.ceil(a / x)
            if x * y < a + hoge:
                continue

            # 現在の比率と9:16の比率の差を計算
            current_diff = abs((x / y) - (9 / 16))

            if current_diff < best_diff:
                best_diff = current_diff
                best_x, best_y = x, y

    return best_x, best_y

# 例
a = 1
x, y = find_closest_ratio(a)
print(f"Rows: {x}, Columns: {y}")

# find_subplot_shape(15)
