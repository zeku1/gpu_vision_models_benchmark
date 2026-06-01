import os
import cv2
import time
import torch
import numpy as np
import multiprocessing
from ultralytics import YOLO

os.environ['FLAGS_enable_pir_api'] = '0'

SAMPLES_DIR = "../samples_night"
VIDEO_FILES = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp4')]

YOLO8_MODEL_PATH = "../yolov8n.pt"
YOLO10_MODEL_PATH = "../yolov10n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7] 
PERSON_CLASSES = [0] 

def processing_worker(cam_idx, video_path, output_queue, stop_event):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo8 = YOLO(YOLO8_MODEL_PATH).to(device)
    yolo10 = YOLO(YOLO10_MODEL_PATH).to(device)
    cap = cv2.VideoCapture(video_path)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        start = time.time()
        v8_res = yolo8(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        v10_res = yolo10(frame, classes=PERSON_CLASSES, verbose=False)[0]
        latency = (time.time() - start) * 1000
        
        for box in v8_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        for box in v10_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        
        cv2.putText(frame, f"CAM {cam_idx} | {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if output_queue.full():
            try: output_queue.get_nowait()
            except: pass
        output_queue.put(frame)
    cap.release()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    stop_event = multiprocessing.Event()
    queues = [multiprocessing.Queue(maxsize=1) for _ in VIDEO_FILES]
    processes = []
    
    for i, path in enumerate(VIDEO_FILES):
        p = multiprocessing.Process(target=processing_worker, args=(i, path, queues[i], stop_event))
        p.start()
        processes.append(p)

    cv2.namedWindow("Night Visualizer Mode B (Multiprocessing)", cv2.WINDOW_NORMAL)
    while True:
        frames = []
        for q in queues:
            try: frames.append(q.get(timeout=0.01))
            except: frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
        h, w = 360, 480
        resized = [cv2.resize(f, (w, h)) for f in frames]
        while len(resized) < 4: resized.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        canvas = np.vstack((np.hstack(resized[:2]), np.hstack(resized[2:4])))
        cv2.imshow("Night Visualizer Mode B (Multiprocessing)", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    stop_event.set()
    for p in processes: p.join()
    cv2.destroyAllWindows()
