import os
import cv2
import time
import torch
import numpy as np
import threading
from ultralytics import YOLO

os.environ['FLAGS_enable_pir_api'] = '0'

SAMPLES_DIR = "../samples"
VIDEO_FILES = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp4')]
if not VIDEO_FILES:
    print("Error: No video samples found in ../samples")
    exit(1)

YOLO8_MODEL_PATH = "../yolov8n.pt"
YOLO10_MODEL_PATH = "../yolov10n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7] 
PERSON_CLASSES = [0] 

class VisionPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.yolo8 = YOLO(YOLO8_MODEL_PATH).to(device)
        self.yolo10 = YOLO(YOLO10_MODEL_PATH).to(device)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo8(dummy, verbose=False)
        self.yolo10(dummy, verbose=False)

    def draw_on_frame(self, frame, cam_idx):
        start = time.time()
        v8_res = self.yolo8(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        v10_res = self.yolo10(frame, classes=PERSON_CLASSES, verbose=False)[0]
        latency = (time.time() - start) * 1000
        
        for box in v8_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        for box in v10_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        
        cv2.putText(frame, f"CAM {cam_idx} | {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

if __name__ == "__main__":
    pipeline = VisionPipeline()
    caps = [cv2.VideoCapture(p) for p in VIDEO_FILES]
    stop_event = threading.Event()
    latest_frames = [None] * len(caps)
    lock = threading.Lock()

    def reader(idx, cap):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with lock:
                latest_frames[idx] = frame

    threads = [threading.Thread(target=reader, args=(i, c)) for i, c in enumerate(caps)]
    for t in threads: t.start()

    cv2.namedWindow("Day Visualizer Mode A (Batched Threads)", cv2.WINDOW_NORMAL)
    while True:
        current_batch = []
        with lock:
            if all(f is not None for f in latest_frames):
                current_batch = [f.copy() for f in latest_frames]
        
        if current_batch:
            processed = []
            for i, frame in enumerate(current_batch):
                processed.append(pipeline.draw_on_frame(frame, i))
            
            h, w = 360, 480
            resized = [cv2.resize(f, (w, h)) for f in processed]
            while len(resized) < 4: resized.append(np.zeros((h, w, 3), dtype=np.uint8))
            
            canvas = np.vstack((np.hstack(resized[:2]), np.hstack(resized[2:4])))
            cv2.imshow("Day Visualizer Mode A (Batched Threads)", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    stop_event.set()
    for t in threads: t.join()
    cv2.destroyAllWindows()
