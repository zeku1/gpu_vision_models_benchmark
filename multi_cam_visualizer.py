import os
import cv2
import time
import torch
import numpy as np
import threading
import queue
from ultralytics import YOLO

os.environ['FLAGS_enable_pir_api'] = '0'

SAMPLES_DIR = "samples"
VIDEO_FILES = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp4')]
if not VIDEO_FILES:
    print("Error: No video samples found.")
    exit(1)

YOLO8_MODEL_PATH = "yolov8n.pt"
YOLO10_MODEL_PATH = "yolov10n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7] 
PERSON_CLASSES = [0] 

display_queues = [queue.Queue(maxsize=1) for _ in VIDEO_FILES]
stop_event = threading.Event()


def processing_worker(cam_idx, video_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Cam {cam_idx}: Starting on {device}")
    
    cap = cv2.VideoCapture(video_path)
    yolo8 = YOLO(YOLO8_MODEL_PATH).to(device)
    yolo10 = YOLO(YOLO10_MODEL_PATH).to(device)
    

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo8(dummy, verbose=False)
    yolo10(dummy, verbose=False)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        start_time = time.time()
        
        v8_res = yolo8(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        v10_res = yolo10(frame, classes=PERSON_CLASSES, verbose=False)[0]
        
        latency = (time.time() - start_time) * 1000
        
        for box in v8_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().item()
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Veh {conf:.2f}", (b[0], b[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        for box in v10_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().item()
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Per {conf:.2f}", (b[0], b[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"CAM {cam_idx} | {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        try:
   
            if display_queues[cam_idx].full():
                display_queues[cam_idx].get_nowait()
            display_queues[cam_idx].put_nowait(frame)
        except queue.Full:
            pass
            
    cap.release()


if __name__ == "__main__":
    threads = []
    for i, path in enumerate(VIDEO_FILES):
        t = threading.Thread(target=processing_worker, args=(i, path))
        t.start()
        threads.append(t)

    print("Starting Tiled Visualizer. Press 'q' to quit.")
    cv2.namedWindow("Real-time Multi-Camera Dashboard", cv2.WINDOW_NORMAL)

    while True:
        frames = []
        for q in display_queues:
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        h, w = 360, 480
        resized = [cv2.resize(f, (w, h)) for f in frames]
        
        if len(resized) < 4:
            for _ in range(4 - len(resized)):
                resized.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        row1 = np.hstack((resized[0], resized[1]))
        row2 = np.hstack((resized[2], resized[3]))
        canvas = np.vstack((row1, row2))

        cv2.imshow("Real-time Multi-Camera Dashboard", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    for t in threads:
        t.join()
    cv2.destroyAllWindows()
