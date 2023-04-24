EXP_NAME="test_dev"
SPLIT_ID=1

# N_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
N_GPUS=1
export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES=4,5,6,7


IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=checkpoints/voc/test/teacher_base/defrcn_det_r101_base1/model_final.pth
# IMAGENET_PRETRAIN=checkpoints/voc/singleHeadAtt_Text/teacher_base/defrcn_det_r101_base1/model_final.pth
# train teacher model
SAVE_DIR=checkpoints/voc/${EXP_NAME}
TEACHER_PATH=checkpoints/voc/${EXP_NAME}/teacher_base/defrcn_det_r101_base${SPLIT_ID}

cfg_MODEL="
MODEL.ROI_HEADS.NAME SematicRes5ROIHeads
MODEL.ROI_HEADS.TEACHER_TRAINING True
MODEL.ROI_HEADS.STUDENT_TRAINING False
MODEL.ROI_HEADS.DISTILLATE False
SOLVER.IMS_PER_BATCH 4

"
# SOLVER.BASE_LR 0.01

python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
       OUTPUT_DIR ${TEACHER_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}
exit

python3 tools/model_surgery.py --dataset voc --method randinit                                \
   --src-path ${TEACHER_PATH}/model_final.pth                    \
   --save-dir ${TEACHER_PATH}

