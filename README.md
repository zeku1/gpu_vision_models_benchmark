# GPU Vision Models Benchmark

This project is a benchmarking suite designed to evaluate the architectural limits of processing 5 simultaneous video streams using **YOLOv8n** and **YOLO10n**. The code is structured to highlight how different concurrency patterns affect real-world FPS.

## Code-Level Constraints & Bottlenecks

The following limitations are inherent to the current implementation:

### 1. Dual-Model Overhead
Every processed frame undergoes **two separate inference passes**:
*   YOLOv8 for vehicle detection.
*   YOLO10 for person detection.
This doubling of computation per frame is the primary bottleneck, typically resulting in **8-15 FPS** depending on the concurrency mode and GPU load.

### 2. Architectural Bottlenecks

*   **Mode A (Sequential Batching):** 
    *   **Constraint:** The code waits for all 5 cameras to provide a frame, then processes them one by one.
    *   **Result:** FPS is limited to `1 / (Inference_Time * 5)`. For a 20ms inference time, this caps the throughput at ~10 FPS.
*   **Mode B (Process Isolation):**
    *   **Constraint:** Spawns 5 independent processes, each with its own CUDA context and model instances.
    *   **Result:** Limited by GPU VRAM and scheduling overhead. While "parallel," GPU resource contention often keeps actual throughput around **12-15 FPS** for the total system.
*   **Mode C (Fixed Worker Pool):**
    *   **Constraint:** Uses a fixed number of worker threads (defaulting to **2 or 3** workers in the code) to process a shared task queue of 5 streams.
    *   **Result:** Throughput is limited by the worker-to-stream ratio. With only 2 workers for 5 streams, the system naturally throttles to ~8-12 FPS per stream.

## Visualizer Layout

The visualizers use a **2x3 grid** (5 active feeds + 1 empty slot) and a 1ms `waitKey` delay. They are designed for real-time monitoring rather than maximum throughput benchmarking.

## How to Run

### Benchmarking
```bash
python Day/benchmark.py
python Night/benchmark.py
```

### Visualization (2x3 Grid)
```bash
python Day/visualizer_b.py
```

## Requirements
*   Python 3.10+
*   PyTorch (CUDA 11.8+)
*   Ultralytics YOLO
*   OpenCV
