# 步驟 1: 匯入所有必要的函式庫
from google.colab import files
import google.colab.patches as colab_pa
import cv2
import numpy as np
import sqlite3 # 匯入資料庫函式庫
# 步驟 2: 影像讀取與前處理 (Colab 環境)
print("請上傳 'p1.jpg' 影像檔案...")
files.upload() # 觸發 Colab 檔案上傳介面

image_path = 'p1.jpg'
image_src = cv2.imread(image_path)

if image_src is None:
    print(f"錯誤：無法讀取影像檔案 {image_path}，請確認檔案已成功上傳。")
    exit()
image = image_src.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 步驟 3: OpenCV 人臉偵測 (Python 程式碼區塊)
casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc_path)
# 執行偵測
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print(f"\n偵測完成。共找到 {len(faces)} 張人臉。")
# 1. 建立或連接到資料庫
db_name = 'face_detection_results.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor() 
sql_create_table = """
CREATE TABLE IF NOT EXISTS face_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    x_coord INTEGER,
    y_coord INTEGER,
    width INTEGER,
    height INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
cursor.execute(sql_create_table)
conn.commit()
print(f"已建立資料庫表格：{db_name}")
# 3. 遍歷偵測結果並插入資料庫 (使用 Cursor 執行 SQL 語句)
sql_insert_data = """
INSERT INTO face_detections (image_name, x_coord, y_coord, width, height) 
VALUES (?, ?, ?, ?, ?);
"""
for (x, y, w, h) in faces:
    # 📌 這裡就是 Cursor 的用途：執行 SQL 語句，將 Python 變數值寫入資料庫
    cursor.execute(sql_insert_data, (image_path, x, y, w, h))
    
    # 同時在影像上繪圖 (這部分仍是 Python/OpenCV 邏輯)
    cv2.rectangle(image,(x,y),(x+w, y+h),(128,255,0),2)
# 4. 提交更改並關閉游標和連接
conn.commit()
cursor.close()
conn.close()
print("所有偵測結果已成功寫入資料庫。")
# 步驟 5: 顯示結果 (Python 程式碼區塊)
imgheight=image.shape[0]
cv2.rectangle(image, (10,imgheight-20), (110,imgheight), (0,0,0), -1)
cv2.putText(image,"Find " + str(len(faces)) + " face!", (10,imgheight-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
colab_pa.cv2_imshow(image)