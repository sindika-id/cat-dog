import time, statistics, numpy as np
import torch
from torch import nn
from torchvision import models
import onnxruntime as ort

# -------------------------
# Config
# -------------------------
ONNX_PATH = "models/catdog_resnet18.onnx"
BATCH_SIZES = [1, 8, 32]      # change as needed
WARMUP = 10
RUNS = 100                    # increase for more stable stats
NUM_THREADS = 4               # tune for your CPU; try 1, 4, or None

# Optional: reduce inter-run noise by fixing threads
if NUM_THREADS is not None:
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(max(1, NUM_THREADS // 2))

# -------------------------
# Load PyTorch model (CPU)
# -------------------------
pt_model = models.resnet18(weights=None)
pt_model.fc = nn.Linear(pt_model.fc.in_features, 2)
pt_model.load_state_dict(torch.load("models/catdog_model.pth", map_location="cpu"))
pt_model.eval()

# -------------------------
# Prepare ONNX Runtime session
# -------------------------
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# so.intra_op_num_threads = NUM_THREADS or leave default
# so.inter_op_num_threads = max(1, (NUM_THREADS or 1)//2)
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])

def bench_pytorch(batch, h=224, w=224):
    x = torch.randn(batch, 3, h, w, dtype=torch.float32)
    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = pt_model(x)
    # measure
    times = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = pt_model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
    return np.array(times)

def bench_onnx(batch, h=224, w=224):
    x = np.random.randn(batch, 3, h, w).astype(np.float32)
    # warmup
    for _ in range(WARMUP):
        _ = sess.run(None, {"input": x})
    # measure
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = sess.run(None, {"input": x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return np.array(times)

def summarize(times_ms, batch):
    mean = float(times_ms.mean())
    p50  = float(np.percentile(times_ms, 50))
    p90  = float(np.percentile(times_ms, 90))
    p99  = float(np.percentile(times_ms, 99))
    thr  = (batch / (mean / 1000.0))  # imgs/s using mean latency
    return mean, p50, p90, p99, thr

print(f"Threads: torch={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
print(f"ONNX Runtime EPs: {sess.get_providers()}")

for b in BATCH_SIZES:
    pt_times = bench_pytorch(b)
    onnx_times = bench_onnx(b)

    pt_stats = summarize(pt_times, b)
    onnx_stats = summarize(onnx_times, b)

    print(f"\nBatch {b}")
    print("PyTorch  : mean={:.2f} ms  p50={:.2f}  p90={:.2f}  p99={:.2f}  thr={:.1f} img/s"
          .format(*pt_stats))
    print("ONNX RT  : mean={:.2f} ms  p50={:.2f}  p90={:.2f}  p99={:.2f}  thr={:.1f} img/s"
          .format(*onnx_stats))
