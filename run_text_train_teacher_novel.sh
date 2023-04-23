EXP_NAME="singleHeadAtt_Text"
SPLIT_ID=1

# N_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
N_GPUS=4
# export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=4,5,6,7


IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=checkpoints/voc/singleHeadAtt_Text/teacher_base/defrcn_det_r101_base1/model_final.pth
# train teacher model
SAVE_DIR=checkpoints/voc/${EXP_NAME}
BASE_DIR=${SAVE_DIR}/teacher_base/defrcn_det_r101_base${SPLIT_ID}

# python3 tools/model_surgery.py --dataset voc --method randinit    \
#    --src-path ${TEACHER_PATH}/model_final.pth                    \
#    --save-dir ${TEACHER_PATH}

BASE_WEIGHT=${BASE_PATH}/model_reset_surgery.pth
for shot in 1 # 1 2 3 5 10  # if final, 10 -> 1 2 3 5 10
do
   for seed in  10 1 2 #3 4 5 6 7 8 9 # 10
   do
   python3 tools/create_config.py --dataset voc --config_root configs/voc               \
      --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
   CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
   
   TEACHER_NOVEL_DIR=${SAVE_DIR}/teacher_novel${SPLIT_ID}/${shot}shot_seed${seed}

   NOVEL_WEIGHT=${BASE_WEIGHT}
   # teacher novel fine-tuning
   cfg_MODEL="
   MUTE_HEADER True
   MODEL.META_ARCHITECTURE GeneralizedRCNN2
   MODEL.ROI_HEADS.NAME TextRes5ROIHeads
   MODEL.ROI_HEADS.TEACHER_TRAINING True
   MODEL.ROI_HEADS.STUDENT_TRAINING False
   MODEL.ROI_HEADS.DISTILLATE False
   SOLVER.IMS_PER_BATCH 16
   "
   # TEST.EVAL_PERIOD 5000
   python3 main.py --num-gpus ${N_GPUS} --config-file ${CONFIG_PATH}                            \
      --opts MODEL.WEIGHTS ${NOVEL_WEIGHT} TEACHER_NOVEL_DIR ${TEACHER_NOVEL_DIR}                     \
         TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

    python3 tools/model_surgery.py --dataset voc --method reset                            \
   --src-path ${TEACHER_NOVEL_DIR}/model_final.pth                    \
   --save-dir ${TEACHER_NOVEL_DIR}

   TEACHER_NOVEL_WEIGHT=${TEACHER_NOVEL_DIR}/model_reset_optimizer.pth
   # student novel distillation
   cfg_MODEL="
   MUTE_HEADER True
   MODEL.META_ARCHITECTURE GeneralizedRCNN2
   MODEL.ROI_HEADS.NAME TextRes5ROIHeads
   MODEL.ROI_HEADS.TEACHER_TRAINING False
   MODEL.ROI_HEADS.STUDENT_TRAINING True
   MODEL.ROI_HEADS.DISTILLATE True
   MODEL.ROI_HEADS.L2 False
   MODEL.ROI_HEADS.KL_TEMP 5
   SOLVER.IMS_PER_BATCH 16
   "
   STUDENT_DIR=${SAVE_DIR}/student_novel${SPLIT_ID}/${shot}shot_seed${seed}

   python3 main.py --num-gpus ${N_GPUS} --config-file ${CONFIG_PATH}                            \
      --opts MODEL.WEIGHTS ${NOVEL_WEIGHT} OUTPUT_DIR ${STUDENT_DIR}                     \
         TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL} 

   rm ${CONFIG_PATH}
   done
done