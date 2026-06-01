import cv2
import os
import glob
import threading
import time
from ultralytics import YOLO
from deepface import DeepFace

RTSP_URL = "rtsp://mficam:mfi122101@192.168.0.121:554/stream2"
EMPLOYEE_DB = "employees/"
FACE_MODEL = "Buffalo_L" 
TARGET_FPS = 10 
FRAME_DELAY = 1.0 / TARGET_FPS 

for pkl_file in glob.glob(os.path.join(EMPLOYEE_DB, "*.pkl")):
    try: os.remove(pkl_file)
    except: pass

yolo_model = YOLO("yolov10n.pt") 

try: 
    DeepFace.find(img_path="dummy", db_path=EMPLOYEE_DB, model_name=FACE_MODEL, enforce_detection=False)
except: 
    pass 

current_identity = "Scanning..."
is_scanning = False 

def scan_face_background(frame_copy):
    global current_identity, is_scanning
    try:
        dfs = DeepFace.find(img_path=frame_copy, db_path=EMPLOYEE_DB, model_name=FACE_MODEL, enforce_detection=False)
        if len(dfs) > 0 and len(dfs[0]) > 0:
            name = os.path.basename(dfs[0].iloc[0]['identity']).split('.')[0]
            current_identity = f"Employee: {name}"
        else:
            current_identity = "Unknown Person"
    except Exception:
        current_identity = "Scanning..."
    is_scanning = False 

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    exit()

cv2.namedWindow(f"Tapo - {FACE_MODEL}", cv2.WINDOW_NORMAL)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    if (current_time - prev_time) < FRAME_DELAY:
        continue 
        
    prev_time = current_time

    results = yolo_model(frame, conf=0.60, classes=[0], verbose=False)
    res_image = frame.copy()

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(res_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(res_image, current_identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        if not is_scanning:
            is_scanning = True
            threading.Thread(target=scan_face_background, args=(frame.copy(),)).start()

    res_image = cv2.resize(res_image, (1280, 720))
    cv2.imshow(f"Tapo - {FACE_MODEL}", res_image)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()