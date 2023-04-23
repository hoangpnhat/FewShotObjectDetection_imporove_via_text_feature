# EXP_NAME="semantic_rpn_roi_heads_x1.2_bbox"
EXP_NAME="distillate_rpn_attn_roi_heads"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


cfg_MODEL="
   MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
   MODEL.ROI_HEADS.NAME TextRes5ROIHeads
   MODEL.ROI_HEADS.KL_TEMP 5
"

python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}

exit