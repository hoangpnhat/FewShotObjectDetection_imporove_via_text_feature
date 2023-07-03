# EXP_NAME="semantic_rpn_roi_heads_x1.2_bbox"
EXP_NAME="distillate_res2_rpn_attn_roi_heads"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


cfg_MODEL="
   MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
   MODEL.ROI_HEADS.NAME SematicRes5ROIHeadsCrossOutput
   MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNAttentionOutputLayers
   MODEL.BACKBONE.FREEZE_AT 1
   MODEL.RESNETS.OUT_FEATURES ['res2','res4']
   MODEL.ADDITION.NAME glove
   SOLVER.IMS_PER_BATCH 12
   SOLVER.MAX_ITER 20000

"

python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}

exit