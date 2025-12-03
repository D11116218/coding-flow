# 人臉偵測 face detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取本地圖片檔案（請將 'abc.jpg' 改為你的圖片路徑）
image_path = 'abc.jpg'
# image_path = 'Otani3.png'  # 另一張圖

image_src = cv2.imread(image_path)

# 檢查圖片是否讀取成功 (建議檢查，避免後續報錯)
if image_src is None:
    print("錯誤：無法讀取影像檔案。")
    print(f"請確認檔案路徑 '{image_path}' 是否正確。")
    exit()  # 終止程式

# 複製一份避免原始圖片被破壞
image = image_src.copy()

# 將圖片轉換為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 讀取人臉偵測的特徵檔
casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc_path)
faces = faceCascade.detectMultiScale(
    gray,  # 最佳做法：在灰階圖上偵測
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20),
    flags=cv2.CASCADE_SCALE_IMAGE
)

imgheight = image.shape[0]  # 圖片高度
imgwidth = image.shape[1]  # 圖片寬度

print('\n--- 偵測結果 ---')
print(f'共偵測到 {len(faces)} 張人臉。')

# 所有的座標輸出必須放在這裡
for i, (x, y, w, h) in enumerate(faces):  # 臉部矩形
    # 輸出單獨人臉的座標資訊
    print(f'\n人臉 {i+1} 資訊:')
    print(f'  長方形座標 (x, y, 寬w, 高h): ({x}, {y}, {w}, {h})')
    print(f'  左上點座標: ({x}, {y})')
    print(f'  右下點座標: ({x+w}, {y+h})')
    # 在臉部區域畫出框框 (這裡使用 x, y, w, h 是安全的)
    cv2.rectangle(image, (x, y), (x+w, y+h), (128, 255, 0), 2)

# 在左下角繪製文字和矩形
cv2.rectangle(image, (10, imgheight-20), (110, imgheight), (0, 0, 0), -1)  # 左下角黑色矩形
cv2.putText(image, "Find " + str(len(faces)) + " face!", (10, imgheight-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 使用 matplotlib 顯示圖片（OpenCV 使用 BGR，matplotlib 使用 RGB，需要轉換）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.axis('off')  # 隱藏座標軸
plt.title('人臉偵測結果')
plt.show()

# 或者使用 OpenCV 的方式顯示（在 Windows 上可能需要額外設定）
# cv2.imshow('Face Detection', image)
# cv2.waitKey(0)  # 等待按鍵
# cv2.destroyAllWindows()  # 關閉視窗

