# # Some basic setup:
# # Setup detectron2 logger
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# # import some common libraries
# import numpy as np
# import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# # import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog


# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/hoangpn/Ecai/DeFRCN/checkpoints/voc/SingleHeadAtt_VKV/teacher_base/defrcn_det_r101_base1/model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)




import pickle
import sys
import numpy as np

import torch 
from torch import nn

from torchnlp.word_to_vector import GloVe

import detectron2
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.layers import ShapeSpec, cat

from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup

# print(MetadataCatalog.get("coco14_test_base"))
# dataset = "voc_2007_test"

# dicts = list(DatasetCatalog.get(dataset))
# metadata = MetadataCatalog.get(dataset)
# # print("dicts: ",dicts)
# print("meta: ",metadata)
# from defrcn.modeling.roi_heads.modules import *

def extender_net(input_size, output_size=2048, factor = 2):
    model = nn.Sequential(
        nn.Linear(input_size, input_size * factor),
        nn.ReLU(),
        nn.Linear(input_size * factor, input_size * factor * factor),
        nn.ReLU(),
        nn.Linear(input_size * factor * factor, output_size),
        nn.ReLU(),
    )
    
    return model
def get_feature_backbone():
    with open('feature_backbone.pkl', 'rb') as f:
        feature_backbone = pickle.load(f)

    return feature_backbone   
def get_image():
    with open('image.pkl', 'rb') as f:
        image = pickle.load(f)

    return image   
def get_proposals():
    with open('proposal.pkl', 'rb') as f:
        proposals = pickle.load(f)
    
    return proposals


def get_gt_classes():
    with open('proposal.pkl', 'rb') as f:
        proposals = pickle.load(f)
        
    gt_classes = cat([p.gt_classes for p in proposals], dim=0)
    return gt_classes
    
def base_class_mapper(text_dim=300):
    glove_vec = GloVe(name='6B', dim=text_dim)
    voc_map = {'aeroplane': 'aeroplane', 'bicycle': 'bicycle', 'boat': 'boat', 'bottle': 'bottle', 
              'car': 'car', 'cat': 'cat', 'chair': 'chair', 'diningtable': 'dining table', 
              'dog': 'dog', 'horse': 'horse', 'person': 'person', 'pottedplant': 'potted plant', 
              'sheep': 'sheep', 'train': 'train', 'tvmonitor': 'tvmonitor', 'bird': 'bird', 
              'bus': 'bus', 'cow': 'cow', 'motorbike': 'motorbike', 'sofa': 'sofa'}
    text_embed = torch.zeros(len(voc_map), text_dim)
    for idx, extend_class in enumerate(voc_map.values()):
        tokens = extend_class.split(' ')
        for token in tokens:
            text_embed[idx] += glove_vec[token]
        text_embed[idx] /= len(tokens)

    return text_embed

def get_feature_pooled():
    with open('feature_pooled.pkl', 'rb') as f:
        feature_pooled = pickle.load(f)
    
    return feature_pooled

# text_embed = base_class_mapper()
# extender = extender_net(input_size=300, output_size=2048)
# extended_embed = extender(text_embed)

# gt_classes = get_gt_classes()
# class_embeds = extended_embed[gt_classes]
# print(class_embeds.shape)

# feature_pooled = get_feature_pooled()
# print(feature_pooled.shape)
import sys
image_feature1 =get_feature_backbone()["res4"][0]
images = get_image()
torch.set_printoptions(threshold=10_000)
# import matplotlib.pyplot as plt
print("feature map",image_feature1.shape)
# np.savetxt('my_file.txt', image_feature1[0].cpu().detach().numpy())
for img in images:
    print(img.shape)
# plt.imsave("image_feature1.jpg",image_feature1[0].cpu().detach().numpy())
# print(image1[0].numpy())
# lv_model = LVModel()
# lv_model.cuda()
# lv_vector = lv_model(proposals, feature_pooled)
# print(proposals)


