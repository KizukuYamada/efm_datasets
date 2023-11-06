import cv2

# YOLOv8を回して車の位置を取得

# 車の位置にバウンディングボックスを描画
# 画像を読み込む
image = cv2.imread('rgb.png')

# バウンディングボックスの座標を定義する (x, y, width, height)
x, y, w, h = 100, 100, 50, 50  # 例として、(100, 100)の位置に幅50、高さ50のボックスを設定

# バウンディングボックスを描画する
# cv2.rectangle()関数には、左上の座標と右下の座標を指定する
cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 赤色で太さ2の線

# 画像を表示する
cv2.imshow('Image with Bounding Box', image)

# 'q'キーが押されたらウィンドウを閉じる
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# # すべてのウィンドウを閉じる
# cv2.destroyAllWindows()

# 車の位置を元に、depthを算出