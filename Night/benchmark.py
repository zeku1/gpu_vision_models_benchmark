import os
import cv2
import time
import torch
import numpy as np
import threading
import queue
import multiprocessing
from ultralytics import YOLO

os.environ['FLAGS_enable_pir_api'] = '0'

SAMPLES_DIR = "../samples_night"
VIDEO_FILES = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp4')]
if not VIDEO_FILES:
    print(f"Error: No video samples found in {SAMPLES_DIR}")
    exit(1)

YOLO8_MODEL_PATH = "../yolov8n.pt"
YOLO10_MODEL_PATH = "../yolov10n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7] 
PERSON_CLASSES = [0] 

class VisionPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Initializing Night VisionPipeline on {device}...")
        self.device = device
        self.yolo8 = YOLO(YOLO8_MODEL_PATH).to(device)
        self.yolo10 = YOLO(YOLO10_MODEL_PATH).to(device)
        
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo8(dummy, verbose=False)
        self.yolo10(dummy, verbose=False)
        
    def process_frame(self, frame):
        metrics = {}
        start = time.time()
        v8_results = self.yolo8(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        v8_conf = v8_results.boxes.conf.cpu().numpy() if len(v8_results.boxes) > 0 else []
        metrics['v8_count'] = len(v8_results.boxes)
        metrics['v8_avg_conf'] = np.mean(v8_conf) if len(v8_conf) > 0 else 0
        
        v10_results = self.yolo10(frame, classes=PERSON_CLASSES, verbose=False)[0]
        v10_conf = v10_results.boxes.conf.cpu().numpy() if len(v10_results.boxes) > 0 else []
        metrics['v10_count'] = len(v10_results.boxes)
        metrics['v10_avg_conf'] = np.mean(v10_conf) if len(v10_conf) > 0 else 0
        
        metrics['latency'] = (time.time() - start) * 1000 
        return metrics

def run_batched_mode(video_paths, duration=30):
    print("\n--- Running Night Mode A: Threaded + Batched Inference ---")
    caps = [cv2.VideoCapture(p) for p in video_paths]
    pipeline = VisionPipeline()
    
    stop_event = threading.Event()
    frames = [None] * len(caps)
    lock = threading.Lock()
    
    def reader_thread(idx, cap):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                continue
            with lock:
                frames[idx] = frame

    threads = []
    for i, c in enumerate(caps):
        t = threading.Thread(target=reader_thread, args=(i, c))
        t.start()
        threads.append(t)

    total_frames = 0
    start_time = time.time()
    all_metrics = []

    while time.time() - start_time < duration:
        batch = []
        with lock:
            if all(f is not None for f in frames):
                batch = [f.copy() for f in frames]
        
        if batch:
            for frame in batch:
                m = pipeline.process_frame(frame)
                all_metrics.append(m)
            total_frames += len(batch)

    stop_event.set()
    for t in threads: t.join()
    for c in caps: c.release()
    
    return total_frames, time.time() - start_time, all_metrics

def worker_process(video_path, result_queue, duration):
    os.environ['FLAGS_enable_pir_api'] = '0' 
    cap = cv2.VideoCapture(video_path)
    pipeline = VisionPipeline()
    start_time = time.time()
    count = 0
    metrics = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        m = pipeline.process_frame(frame)
        metrics.append(m)
        count += 1
    
    cap.release()
    result_queue.put((count, metrics))

def run_multiprocessing_mode(video_paths, duration=30):
    print("\n--- Running Night Mode B: Multiprocessing (Independent) ---")
    result_queue = multiprocessing.Queue()
    processes = []
    
    start_time = time.time()
    for path in video_paths:
        p = multiprocessing.Process(target=worker_process, args=(path, result_queue, duration))
        p.start()
        processes.append(p)
    
    total_frames = 0
    all_metrics = []
    for _ in range(len(video_paths)):
        count, metrics = result_queue.get()
        total_frames += count
        all_metrics.extend(metrics)
        
    for p in processes: p.join()
    return total_frames, time.time() - start_time, all_metrics

def run_queue_mode(video_paths, duration=30):
    print("\n--- Running Night Mode C: Thread-based Pipeline with Queues ---")
    frame_queue = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    
    def producer(path):
        cap = cv2.VideoCapture(path)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue
        cap.release()

    pipeline = VisionPipeline()
    processed_count = [0]
    all_metrics = []
    
    def consumer():
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)
                m = pipeline.process_frame(frame)
                all_metrics.append(m)
                processed_count[0] += 1
            except queue.Empty:
                continue

    producers = [threading.Thread(target=producer, args=(p,)) for p in video_paths]
    consumers = [threading.Thread(target=consumer) for _ in range(3)] 
    
    start_time = time.time()
    for t in producers + consumers: t.start()
    
    time.sleep(duration)
    stop_event.set()
    
    for t in producers + consumers: t.join()
    
    return processed_count[0], time.time() - start_time, all_metrics

def print_report(name, total_frames, total_time, metrics):
    if not metrics:
        print(f"\nRESULTS FOR {name}: No data collected.")
        return
    fps = total_frames / total_time
    avg_latency = np.mean([m['latency'] for m in metrics])
    print(f"\nRESULTS FOR {name}:")
    print(f"  Total Frames: {total_frames}")
    print(f"  Overall FPS:  {fps:.2f}")
    print(f"  Avg Latency:  {avg_latency:.1f}ms")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    TEST_DURATION = 20 
    
    f_a, t_a, m_a = run_batched_mode(VIDEO_FILES, TEST_DURATION)
    print_report("NIGHT MODE A (Threaded + Batched)", f_a, t_a, m_a)
    
    f_b, t_b, m_b = run_multiprocessing_mode(VIDEO_FILES, TEST_DURATION)
    print_report("NIGHT MODE B (Multiprocessing)", f_b, t_b, m_b)
    
    f_c, t_c, m_c = run_queue_mode(VIDEO_FILES, TEST_DURATION)
    print_report("NIGHT MODE C (Threaded Queues)", f_c, t_c, m_c)
