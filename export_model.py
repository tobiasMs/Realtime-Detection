# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:49:29 2025

@author: tobias.sulistiyo
"""

import torch
from torchvision import models

def export_onnx(weights="runs/mobilenetv2/best.pt", out="runs/mobilenetv2/model.onnx"):
    device="cpu"
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.last_channel, 2)
    m.load_state_dict(torch.load(weights, map_location=device)["model"])
    m.eval()
    x = torch.randn(1,3,224,224)
    torch.onnx.export(m, x, out,
                      input_names=["input"], output_names=["logits"],
                      opset_version=12,
                      dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print("Model exported:", out)

if __name__=="__main__":
    export_onnx()
