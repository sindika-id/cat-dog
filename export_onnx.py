import torch
from torch import nn
from torchvision import models

# 1. Define the model architecture (must match training)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# 2. Load trained weights
ckpt = torch.load("models/catdog_model.pth", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

print("Model loaded and ready for ONNX export.")

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "models/catdog_resnet18.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True
)

print("Exported to catdog_resnet18.onnx")
