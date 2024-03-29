import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec, cat, nonzero_tuple

from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs

from defrcn.data.builtin_meta import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES
from defrcn.data.builtin_meta import _get_coco_fewshot_instances_meta

from .attentive_modules import *
from .my_module import *
from ..meta_arch.gdl import decouple_layer, AffineLayer

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, self.out_channels = self._build_res5_block(cfg)
        self.output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(
            cls_features
        )

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class TextRes5ROIHeads(Res5ROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # self.use_mem = cfg.MODEL.ROI_HEADS.USE_MEMORY
        # self.use_ot = cfg.MODEL.ROI_HEADS.USE_OT
        # self.is_freeze_mpl = cfg.MODEL.ROI_HEADS.FREEZE_MPL
        # self.use_background = True
        # self.repeat_time = cfg.MODEL.ROI_HEADS.REPEATED_TIME
        # self.factors = cfg.MODEL.ROI_HEADS.FACTORS
        # self.capacity = cfg.MODEL.ROI_HEADS.MEM_CAPACITY
        # self.use_bbox = cfg.MODEL.ROI_HEADS.USE_BBX
        # self.meta_loss_weight = cfg.MODEL.META_LOSS_WEIGHT
        # self.syn_loss_weight = cfg.MODEL.SYN_LOSS_WEIGHT
        # self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        # self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        # self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        # self.novel_tuning = True
        # self.save_dir = cfg.OUTPUT_DIR
        # self.student_l2_loss = cfg.MODEL.ROI_HEADS.L2
        # self.student_l2_loss_cosine = cfg.MODEL.ROI_HEADS.L2_COSINE
        # self.student_kl_loss = cfg.MODEL.ROI_HEADS.KL
        # self.student_kl_temp = cfg.MODEL.ROI_HEADS.KL_TEMP

        self.__init_LV_model__(self.out_channels, cfg)

        # super_num_class = self.num_group*self.num_k
        # is_super = False
        # num_class = super_num_class if is_super else self.num_classes
        num_class = self.num_classes

        self.stu_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.tracker_copy_weight = False

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention(
            input_size, cfg=cfg, is_multi=False)
        if self.training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass


    def forward_adapter(self, fg_features, teacher_features=None):

        # feat = self.attention.forward_wo_label(fg_features)
        feat = self.mlp_adapter(fg_features)
        loss = {}
        alpha = 1.0
        margin = 0.2
        def norm_x(x): return F.normalize(x)

        def loss_cosine(a, b):
            return 1 - torch.einsum(
                'b i, b i -> b', norm_x(a), norm_x(b))

        if self.training and self.distill_mode and self.student_l2_loss:
            # l = ((feat - teacher_features)**2).mean()*0

            # l = loss_cosine(feat, teacher_features).mean()*alpha
            # l = (l - margin).clamp(min=0.2)
            # l[torch.where(l < margin)] = 0
            # l = l.mean()*alpha
            # print('feat', feat)
            # print('teacher_features', teacher_features)

            if self.student_l2_loss_cosine:
                l = F.cosine_embedding_loss(feat, teacher_features, Variable(
                    torch.Tensor(feat.size(0)).cuda().fill_(1.0)))
            else:
                l = F.mse_loss(feat, teacher_features)*alpha

            # l = l/feat.shape[0]  # mean according batch size
            loss = {'loss_student_feat': l}

        return feat, loss

    def _get_gt_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

        sampled_idxs = torch.cat([positive, negative], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_proposals(self, proposals, targets):

        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_teacher(self, feature_pooled, proposals, test_with_gt=True):

        gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        num_preds_per_image = [len(p) for p in proposals]
        # loss_att, output_att = self.forward_with_text_attention(
        #     feature_pooled, gt_classes=gt_classes, num_preds_per_image=num_preds_per_image)

        loss_att, output_att = self.attention(feature_pooled, gt_classes, num_preds_per_image)
        
        # loss_att, output_att = self.attention(feature_pooled)
        # print(output_att['sim2stext'].shape)
        # print(output_att.len)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward_student(self, feature_pooled, proposals, teacher_output=None):
        if teacher_output is not None:
            teacher_features = teacher_output.get('sim2stext')
        else:
            teacher_features = None

        att_feature, loss = self.forward_adapter(
            feature_pooled, teacher_features=teacher_features)

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        # if self.student_training and self.training and self.distill_mode and self.student_kl_loss:
        if self.training:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': self.student_kl_temp,  # 1, 5, 10, 15
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward(self, images, features, proposals, targets=None):
        del images
        test_with_gt = True if (not self.training) and targets else False
        # print('test_with_gt:', test_with_gt)
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        t_output = {}
        # if self.teacher_training or (self.student_training and self.training and self.distill_mode):
        if self.training:
            t_output, t_loss = self.forward_teacher(
                feature_pooled, proposals)
            t_loss = {key+'_t': val for key, val in t_loss.items()}

            teacher_outputs = FastRCNNOutputs(
                self.box2box_transform,
                t_output['pred_logits'],
                t_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )
            
            
            s_output, s_loss = self.forward_student(
                feature_pooled, proposals, t_output)

            student_outputs = FastRCNNOutputs(
                self.box2box_transform,
                s_output['pred_logits'],
                s_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )
            
            
            losses = {}
            teacher_loss = teacher_outputs.losses()
            teacher_loss = {key+'_t': val for key, val in teacher_loss.items()}

            losses.update(teacher_loss)
            losses.update(t_loss)

            losses.update(student_outputs.losses())
            losses.update(s_loss)

            return [], losses
        else:
            s_output, s_loss = self.forward_student(feature_pooled, proposals, None)
            pred_instances, _ = student_outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class TextRes5ROIHeads_VKV(TextRes5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention_VKV(
            input_size, cfg=cfg, is_multi=False)
        if self.student_training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass


@ROI_HEADS_REGISTRY.register()
class TextRes5ROIHeads_textDomination(TextRes5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention_textDomination(
            input_size, cfg=cfg, is_multi=False)
        if self.student_training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass


@ROI_HEADS_REGISTRY.register()
class TextRes5ROIHeads_textDomination_VKV(TextRes5ROIHeads_textDomination):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention_textDomination_VKV(
            input_size, cfg=cfg, is_multi=False)
        if self.student_training:
            # self.mlp_adapter = MLP(input_size, widening_factor=2)
            # self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass

@ROI_HEADS_REGISTRY.register()
class SematicRes5ROIHeads(Res5ROIHeads):
    
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.__init_LV_model__(self.out_channels, cfg)
        self.device = 'cuda'
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        # self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
        #     cfg, self.out_channels, 15, self.cls_agnostic_bbox_reg,
        #     # num_super_classes=super_num_class
        # )

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.addition_model = cfg.MODEL.ADDITION.NAME
        if self.addition_model is not None:
            if self.addition_model == "glove":
                self.semantic_dim = 300
            elif self.addition_model == "clip":
                self.semantic_dim = 512
        self.attention = SematicProposalAttention(
            input_size, cfg=cfg, is_multi=False)
        
        if cfg.MODEL.ADDITION.FREEZEATTENTION:
            for p in self.attention.parameters():
                p.requires_grad = False
            print("froze AttentionModule parameters")

            # print(sum([torchtorch.numel(p.data.shape)for p in self.attention.parameters()]))
            # print(.data.shape)
            
            
        self.output_projection = nn.Linear(input_size,self.semantic_dim)
        self.sematic_projection = nn.Linear(self.semantic_dim,input_size)
        self.projection_matrix = nn.Parameter(torch.randn(self.semantic_dim,input_size) * 1e-8)

        
        pass
    def _get_gt_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

        sampled_idxs = torch.cat([positive, negative], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_proposals(self, proposals, targets):

        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def cal_CE_att(self, output_att,gt_classes):
        loss_att = {}
        attentive_feat = self.output_projection(output_att['sim2stext'])
        attentive_feat = F.relu(attentive_feat)
        attentive_score = torch.matmul(attentive_feat, output_att['text_feat'].transpose(0, 1))

        attentive_score = F.softmax(attentive_score,dim =1)

        loss_entropy=F.cross_entropy(
            attentive_score, gt_classes , reduction="mean"
        )
        loss_att['loss_attentive'] = loss_entropy
        # import pdb; pdb.set_trace()

        # threshHold =0.8
        # Guided_gt_classes = gt_classes
        # logits, indices = attentive_score.max(dim=1)
        # for indx,logit in enumerate(logits):
        #     if logit >= threshHold:
        #         Guided_gt_classes[indx] = indices[indx]
        # print("gt_classes",gt_classes)
        # print(Guided_gt_classes)
        return loss_att

    def forward_att(self, feature_pooled,gt_classes=0):
        attentive_score,output_att = self.attention(feature_pooled)
        loss_att ={}

        if self.training:
            # loss_att = self.cal_CE_att(output_att,gt_classes)
            loss_att['loss_attentive'] = F.cross_entropy(
            attentive_score[0], gt_classes , reduction="mean"
        )
        # Guided_gt_classes = self.cal_CE_att(output_att,gt_classes)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        # pred_class_logits, pred_proposal_deltas = self.box_predictor(
        #     feature_pooled)

        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward(self, images, features, proposals, targets=None):
        
        del images
        test_with_gt = True if (not self.training) and targets else False
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        att_output = {}
        
        if self.training:
            del features
            att_output, att_loss = self.forward_att(
                feature_pooled, gt_classes)
            # import pdb; pdb.set_trace()

            del feature_pooled
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                att_output['pred_logits'],
                att_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta
            )
            att_loss = {key: val for key, val in att_loss.items()}
            losses = {}
            loss = outputs.losses()
            losses.update(loss)
            losses.update(att_loss)
            return [], losses
        else:
            att_output, att_loss = self.forward_att(
                feature_pooled)
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                att_output['pred_logits'],
                att_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )   
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            
            return pred_instances, {}
@ROI_HEADS_REGISTRY.register()
class SematicRes5ROIHeadsCrossOutput(SematicRes5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
    def forward_att(self, feature_pooled,gt_classes=0):
        loss_att, output_att = self.attention(feature_pooled)
        
        attentive_feat = self.output_projection(output_att['sim2stext'])
        attentive_feat = F.relu(attentive_feat) 
        attentive_score = torch.matmul(attentive_feat, output_att['text_feat'].transpose(0, 1))
 
        # if self.training:
        #     loss_att['CE_attention_loss']=F.cross_entropy(
        #         attentive_score, gt_classes
        #     )
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, attentive_score)
        # print(pred_class_logits.shape)
        # print(gt_classes.shape)
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att
    
