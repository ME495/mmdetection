import torch
from torch.nn import Module
from torch import Tensor

from mmdet.apis import inference_detector, init_detector

config_path = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_ir_right_left.py'
checkpoint_path = 'work_dirs/rtmdet_tiny_ir_right_left3/epoch_19.pth'
device = 'cuda:0'

class RTMDet(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> tuple:
        results = []
        neck_outputs = self.model(inputs)
        for feats in zip(*neck_outputs):
            results.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
        return tuple(results)


model = init_detector(config_path, checkpoint_path, device=device)
rtmdet = RTMDet(model)
rtmdet.eval()

input = torch.randn(1, 3, 480, 640).to(device)

output = rtmdet(input)

print(output[0].shape, output[1].shape, output[2].shape)

torch.onnx.export(
    rtmdet, input,
    'rtmdet.onnx',
    input_names=['input'],
    opset_version=11,
    dynamic_axes={'input':{0: 'batch'}})