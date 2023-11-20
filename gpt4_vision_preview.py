from openai import OpenAI
import pdb
import base64
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import os
import ast
import re

def extract_numbers_and_format(content):
    # 文字列から数字を見つける
    numbers = re.findall(r'\d+', content)

    # 最初の2つの数字を取得（存在しない場合は9で埋める）
    for i in range(1):
        if i < len(numbers):
            formatted_numbers = int(numbers[i])
        else:
            formatted_numbers = 9 
    # formatted_numbers = [int(numbers[i]) if i < len(numbers) else 9 for i in range(1)]

    # # 末尾に9を追加
    # formatted_numbers.append(9)

    return formatted_numbers

def get_frame_as_base64(video_path, frame_number):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)

    # フレーム番号を指定してフレームを読み込む
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    # フレームが正しく読み込まれたか確認
    if not ret:
        print("フレームの取得に失敗しました。")
        return None

    # OpenCVのBGR形式からRGB形式に変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # PILイメージに変換
    im = Image.fromarray(frame)

    # バイトストリームに保存
    buffered = BytesIO()
    im.save(buffered, format="JPEG")

    # Base64エンコード
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 画像を指定されたパスに保存
    # pdb.set_trace()
    save_directory = os.path.dirname(video_path)
    save_directory = os.path.join(save_directory, "GTP4_vision_preview2")
    save_filename = os.path.join(save_directory, f"frame_{frame_number}.jpg")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    im.save(save_filename)
    
   
    # リソースを解放
    cap.release()
    
    return img_str

# 使用例
# video_path = '/mnt/fsx/datasets/Platooning/20210916_084834000_iOS_short.mp4'
video_path = '/mnt/fsx/datasets/Platooning/20210916_084834000_iOS.mov'
npy_path = 'selected_frames.npy' 
# analysis_num = [78000,79000]
analysis_num = np.load(npy_path)

answer_list= np.zeros((len(analysis_num), 2), dtype=int)
# answer_list[0,2] = 3
for i in range(len(analysis_num)):
    # pdb.set_trace()
    frame_number = analysis_num[i]  # 取得したいフレーム番号
    base64_image = get_frame_as_base64(video_path, frame_number)

    # def encode_image(image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')

    # image_path = "./YOLOv8/examples/rgb_input99_20210916_084834000_iOS_short.png" # 例 /Users/user/images_folder/image.png
    # base64_image = encode_image(image_path)

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": "この画像は、日本の高速道路を進行方向に向かって撮影しているものだよ。前方の白い車は、左から数えて何番目の車線を走行してる？また、この画像は左から数えて何番目の車線から撮影されてる？ここが工事区間かどうかも教えて。回答の形式は全て数字のみで、[白い車の走行車線,撮影車線,工事区間かどうか]でお願いします。工事区間かどうかについては、工事区間だったら1を、そうでなかったら0で回答お願い。もしわからないとか、うまく答えられないときは、その要素の数字を9にしといて。"},
                    # {"type": "text", "text": "この画像は、日本の高速道路を進行方向に向かって撮影しているものだよ。前方の白い車は、左から数えて何番目の車線を走行してる？回答の形式は数字のみでお願いします。もしわからないとか、うまく答えられないときは、その要素の数字を9にしといて。"},
                    {"type": "text", "text": "この画像は、日本の高速道路を進行方向に向かって撮影しているものだよ。この画像は左から数えて何番目の車線から撮影されてる？回答の形式は数字のみでお願いします。もしわからないとか、うまく答えられないときは、その要素の数字を9にしといて。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=100,
    )


    print(response.choices[0])
    # print(response.choices[0].message.content)
    content_str = response.choices[0].message.content
    if len(content_str)>5:
        answer = extract_numbers_and_format(content_str)
    else:
        answer = ast.literal_eval(content_str)
    # pdb.set_trace()
    print(answer)
    # answer_i_list = [frame_number, answer[0], answer[1], answer[2]]
    answer_i_list = [frame_number, answer]
    answer_list[i] = answer_i_list
    # pdb.set_trace()

#answer_listをvido_pathのところに保存
save_directory = os.path.dirname(video_path)
save_directory = os.path.join(save_directory, "GTP4_vision_preview")
save_filename = os.path.join(save_directory, f"answer_list.txt")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
np.savetxt(save_filename, answer_list, fmt='%d')
