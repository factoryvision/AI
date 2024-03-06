import torch
from models.yolo import Model

model = Model(cfg='cfg/training/yolov7.yaml')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "yolov7-w6-pose.onnx")