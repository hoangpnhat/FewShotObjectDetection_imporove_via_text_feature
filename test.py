import pickle
import sys
import numpy as np
from defrcn.config import get_cfg, set_global_cfg
import torch 
from torch import nn

ckpt = torch.load('checkpoints/voc/singleHeadAtt_Text/student_novel1/1shot_seed10/model_final.pth', map_location='cpu')
params = ckpt['model']
print(type(params))
total = sum(p.numel() for p in params.values())
print(total)
# for layer_tensor_name, tensor in tensor_list[0]:
#     print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))