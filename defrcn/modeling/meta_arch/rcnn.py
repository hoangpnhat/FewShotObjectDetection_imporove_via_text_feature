import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads

from defrcn.data.builtin_meta import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES
from defrcn.data.builtin_meta import _get_coco_fewshot_instances_meta
from defrcn.modeling.roi_heads.attentive_modules import *
from defrcn.utils.class_embedding import get_class_embed
import math 
__all__ = ["GeneralizedRCNN", "GeneralizedTextRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        _, _, results, image_sizes = self._forward_once_(batched_inputs, gt_instances)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        
        # for image in processed_results:
        #     print(image['instances'])
        #     print(gt_instances)
        #     print("$"*50)
        #     print(image.shape)
            
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
    
 
 
 
    
@META_ARCH_REGISTRY.register()
class GeneralizedSemanticRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.addition_model = cfg.MODEL.ADDITION.NAME
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.visual_dim = self._SHAPE_["res4"].channels
        
        if self.addition_model is not None:
            if self.addition_model == "glove":
                self.semantic_dim = 300
            elif self.addition_model == "clip":
                self.semantic_dim = 512
    
            self.fixed_bg = False
            self.class_names = get_class_name(cfg)
            self.class_embedding = get_class_embed(
                self.class_names, 
                self.addition_model,
                include_bg=self.fixed_bg 
                )
            if not self.fixed_bg:
                self.bg_feature_init = torch.randn(1, self.semantic_dim)
                self.bg_feature = nn.parameter.Parameter(
                    self.bg_feature_init.clone(), 
                    requires_grad=True
                    )
            
            self.sem2vis_proj = nn.Conv2d(self.semantic_dim, self.visual_dim, kernel_size=1, bias=False)
            
        self.to(self.device)
        
    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if self.addition_model is not None:
            features = {k: self._generate_semantic_features(features[k], gt_instances, self.class_embedding)
                        for k in features}
            
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes
    
    def _expand_bbox(self, gt_box, max_size, stride, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / stride
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, x_c - w/2))
        y1 = int(max(0, y_c - h/2))
        x2 = int(min(max_size[1], x_c + w/2)) 
        y2 = int(min(max_size[0], y_c + h/2))
        return x1, y1, x2, y2
    
    def _generate_semantic_features(self, vis_features, gt_instances, stride, expand_rate):
        max_size = vis_features.shape[-2:]
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        
        features = torch.zeros(
            (len(gt_instances), max_size[0], max_size[1], self.semantic_dim),
            device=self.device
            )
        features[:,:,:] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, max_size, stride, expand_rate)
                features[idx, y1:y2, x1:x2] = self.class_embedding[gt_class]
        
        # (B, H, W, channels) -> (B, channels, H, W)
        features = features.permute(0, 3, 1, 2)
        
        return features

        
@META_ARCH_REGISTRY.register()
class GeneralizedDistillatedRCNN(GeneralizedSemanticRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.vis2sem_proj = nn.ModuleDict()
        for scale in self._SHAPE_:
            self.vis2sem_proj[scale] = nn.Conv2d(
                self._SHAPE_[scale].channels, self.semantic_dim, kernel_size=1,
                ).to(self.device)
        self.to(self.device)
        
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        kd_loss, proposal_losses, detector_losses, _, _ = self._forward_once_(
            batched_inputs, gt_instances
            )
        losses = {}
        losses.update(kd_loss)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    def inference(self, batched_inputs):
        assert not self.training
        _, _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        multi_scale_features = self.backbone(images.tensor)

        kd_loss = {}
        if self.training:
            kd_loss = self._distillate_multi_scale_features(multi_scale_features, gt_instances)
            
        features = {'res4': multi_scale_features['res4']}
        del multi_scale_features
        
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return kd_loss, proposal_losses, detector_losses, results, images.image_sizes
    
    def _distillate_combined_features(self, vis_features, gt_instances, class_embedding, stride=16):        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        max_size = vis_features.shape[-2:]
        
        semantic_features = torch.zeros(
            (len(gt_instances), max_size[0], max_size[1], self.semantic_dim), 
            device=self.device
            )
        if self.fixed_bg:
            semantic_features[:,:,:] = class_embedding[self.num_classes]
        else:
            semantic_features[:,:,:] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, max_size, stride, 1.0)
                semantic_features[idx, y1:y2, x1:x2] = class_embedding[gt_class]

        # vis_features = vis_features.permute(0, 2, 3, 1) # (B, 300, H, W) -> (B, H, W, 1024)
        # kd_loss = F.mse_loss(vis_features, semantic_features)
        
        ### combined
        # (B, H, W, 1024) + (B, H, W, 300) -> (B, H, W, 1324)
        semantic_features = semantic_features.permute(0, 3, 1, 2) # (B, H, W, 300) -> (B, 300, H, W)
        combined_features = torch.cat([vis_features, semantic_features], dim=1) 
        combined_features = self.combined2sem_proj(combined_features)
        vis_features = self.vis2sem_proj(vis_features)
        
        kd_loss = F.mse_loss(vis_features, semantic_features)
        
        return {"loss_kd": kd_loss}
    
    def _distillate_multi_scale_features(self, visual_features, gt_instances):        
        losses = {}
        for scale in visual_features:
            stride = self._SHAPE_[scale].stride
            if scale == 'res2':
                expand_rate = 1.0
            elif scale == 'res3':
                expand_rate = 1.0
            else: # scale == 'res4':
                expand_rate = 1.0
            semantic_features = self._generate_semantic_features(
                visual_features[scale], gt_instances, stride, expand_rate
                ).to(self.device)
            projected_visual_features = self.vis2sem_proj[scale](visual_features[scale])
            
            losses.update({f'loss_rpn_{scale}': F.mse_loss(projected_visual_features, semantic_features)})
        
        return losses
    
@META_ARCH_REGISTRY.register()
class GeneralizedDistillatedAddingRCNN(GeneralizedDistillatedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _generate_semantic_features(self, vis_features, gt_instances, stride, expand_rate):
        max_size = vis_features.shape[-2:]
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        
        semantic_features = torch.ones(
            (len(gt_instances), max_size[0], max_size[1], self.semantic_dim),
            device=self.device
            )
        instance_features =  torch.zeros(
            (len(gt_instances), max_size[0], max_size[1], self.semantic_dim),
            device=self.device
            )
        lst_sematic_features = []
        # if self.fixed_bg:
        #     semantic_features[:,:,:] = self.class_embedding[self.num_classes]
        # else:
        #     semantic_features[:,:,:] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, max_size, stride, 1.0)
                semantic_features[idx, y1:y2, x1:x2] = 0
                instance_features[idx, y1:y2, x1:x2] += self.class_embedding[gt_class]
                
        semantic_features = torch.where(semantic_features==1,self.bg_feature, instance_features)
        # (B, H, W, channels) -> (B, channels, H, W)
        semantic_features = semantic_features.permute(0, 3, 1, 2)
        
        return semantic_features
 
@META_ARCH_REGISTRY.register()
class GeneralizedTextRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rpn_addition = cfg.MODEL.RPN.ADDITION
        self.teacher_training = cfg.MODEL.DISTILLATION.TEACHER_TRAINING

        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.features_channels = self._SHAPE_["res4"].channels
        self.semantic_dim = cfg.ADDITION.SEMANTIC_DIM
        self.to_rpn_input_proj = nn.Linear(self.features_channels + self.semantic_dim, self.features_channels).to(self.device)
        self.class_names = self._get_class_name(cfg)
        self.class_embed = get_class_embed(self.class_names, model="glove", semantic_dim=self.semantic_dim).to(self.device)
        self.bg_feature_init = torch.randn(1, self.semantic_dim)
        self.bg_feature = nn.parameter.Parameter(self.bg_feature_init.clone(), requires_grad=True).to(self.device)

        
    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        features = {k: self._add_semantic_features(features[k], gt_instances, self.class_embed)
                    for k in features}
        
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def _expand_bbox(self, gt_box, max_size, stride, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / stride
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, (x_c - w/2) // 1)) # x // 1 = floor(x)
        y1 = int(max(0, (int(y_c - h/2) // 1)))
        x2 = int(min(max_size[1], (x_c + w/2) // 1 + 1)) # x // 1 = ceil(x)
        y2 = int(min(max_size[0], (y_c + h/2) // 1 + 1))
        return x1, y1, x2, y2

    def _add_semantic_features(self, vis_features, gt_instances, semantic_features, stride=16):
        # (B, channels, H, W) -> (B, H, W, channels)
        vis_features = vis_features.permute(0, 2, 3, 1)
        
        max_size = vis_features.shape[1:3]
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        
        features = torch.zeros((len(gt_instances), max_size[0], max_size[1], self.semantic_dim), 
                               device=self.device)
        features[:,:,:] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, max_size, stride, 1.0)
                # print(x1, y1, x2, y2)
                features[idx, y1:y2, x1:x2] = semantic_features[gt_class]
        
        features = torch.cat((vis_features, features), dim=-1)
        features = self.to_rpn_input_proj.to(self.device)(features)
        
        # (B, H, W, channels) -> (B, channels, H, W)
        features = features.permute(0, 3, 1, 2)
        
        return features

    def _get_class_name(self, cfg):
        dataset = cfg.DATASETS.TRAIN[0]
        if 'voc' in dataset:
            if 'base' in dataset:
                classes = PASCAL_VOC_BASE_CATEGORIES[int(dataset.split('_')[-1][-1])]
            if 'novel' in dataset:
                classes = PASCAL_VOC_NOVEL_CATEGORIES[int(dataset.split('_')[-3][-1])]
            if 'all' in dataset:
                classes = PASCAL_VOC_ALL_CATEGORIES[int(dataset.split('_')[-3][-1])]
        if 'coco' in dataset:
            ret = _get_coco_fewshot_instances_meta()
            if 'base' in dataset:
                classes = ret["base_classes"]
            if 'novel' in dataset:
                classes = ret["novel_classes"]
            if 'all' in dataset:
                classes = ret["thing_classes"]
        return classes


@META_ARCH_REGISTRY.register()
class GeneralizedTextAttRCNN(GeneralizedTextRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.LtoVproj = nn.Linear(self.semantic_dim, self.features_channels).to(self.device)
        self.attproj = nn.Linear(self.features_channels*2,self.features_channels).to(self.device)
        self.attention = SingleHeadSiameseAttention(self.features_channels)
        self.attention.to(self.device)
        
    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
       
        features = {k: self.add_semantic_features(features[k], gt_instances, self.semantic_features)
                    for k in features}
        
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes
    
    def add_semantic_features(self, vis_features, gt_instances, semantic_features, stride=16):
        # (B, channels, H, W) -> (B, H, W, channels)
        vis_features = vis_features.permute(0, 2, 3, 1)
        
        num_classes = semantic_features.shape[0]
        max_size = vis_features.shape[1:3]
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        
        features = torch.zeros((len(gt_instances), max_size[0], max_size[1], self.semantic_dim), 
                               device=self.device) #(B,H,W,300)
        features[:,:,:] = semantic_features[num_classes - 1]
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = (gt_box / stride).int()
                features[idx, x1:x2, y1:y2] = semantic_features[gt_class]
        

        ## ATTENTION
        features = self.LtoVproj(features)

        value_features = torch.cat((vis_features,features),dim=-1).to(self.device)
        value_features =self.attproj(value_features)
        
        q = torch.flatten(vis_features, start_dim = 1,end_dim=2) # (sz_b,H,W,C) --->> (sz_b,HxW,C)
        k = torch.flatten(features, start_dim = 1,end_dim=2)
        v = torch.flatten(value_features, start_dim = 1,end_dim=2)

        att_features = self.attention(q=q,k=k,v=v) # (sz_b,HxW,C)

        att_features = att_features.view(len(gt_instances),max_size[0], max_size[1],self.features_channels) #(sz_b,HxW,C)--->>(sz_b,H,W,C) 

        # (B, H, W, channels) -> (B, channels, H, W)
        att_features = att_features.permute(0, 3, 1, 2)
        return att_features

  