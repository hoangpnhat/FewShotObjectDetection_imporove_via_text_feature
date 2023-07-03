import pickle
import sys
import numpy as np
from defrcn.config import get_cfg, set_global_cfg
import torch 
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ckpt = torch.load('checkpoints/voc/singleHeadAtt_Text/student_novel1/1shot_seed10/model_final.pth', map_location='cpu')
# params = ckpt['model']
# print(type(params))
# total = sum(p.numel() for p in params.values())
# print(total)
# for layer_tensor_name, tensor in tensor_list[0]:
#     print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
gt_classes = torch.load('gt_classes.t')
predict_data = torch.load('attentive_score.t')

gt_classes = gt_classes.tolist()
predict_data = predict_data.tolist()
cf_matrix = confusion_matrix(gt_classes, predict_data)
# cf_matrix = cf_matrix/np.sum(cf_matrix)
# np.fill_diagonal(cf_matrix, -0.1)

cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
# print(cf_matrix)
sns.set(rc = {'figure.figsize':(20, 20)})
ax = sns.heatmap(cf_matrix, annot=True,
            fmt='.2%', cmap='viridis')
ax.set_title('Confusion Matrix with attentive_score\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "background"
    ])
ax.yaxis.set_ticklabels(["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "background"
    ])
## Display the visualization of the Confusion Matrix.
plt.savefig('confusetionMatrixAttentive.png')
