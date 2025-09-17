import torch
import onnxruntime as ort
import numpy as np
from torch import nn
from torchvision import models

# Load PyTorch model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/catdog_model.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)

# PyTorch output
with torch.no_grad():
    pt_out = model(dummy).numpy()

# ONNX Runtime output
session = ort.InferenceSession("models/catdog_resnet18.onnx", providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"input": dummy.numpy()})[0]

# Compare
diff = np.max(np.abs(pt_out - onnx_out))
print("Max abs diff:", diff)