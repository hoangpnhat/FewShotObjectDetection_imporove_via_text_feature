EXP_NAME="branch_attention_fixbackground_negMeanClass_CrossOutputAttention"
# EXP_NAME="test"

SPLIT_ID=1
N_GPUS=2
# export CUDA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES=4,5
# SEED = 44029952

IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=checkpoints/voc/AttentionRoiHead_CE/teacher_base/defrcn_det_r101_base1/model_final.pth
# IMAGENET_PRETRAIN=checkpoints/voc/Pure_attention_fix_background_orthogonalMeanClass/teacher_base/defrcn_det_r101_base1/model_final.pth
# train teacher model
SAVE_DIR=checkpoints/voc/${EXP_NAME}
TEACHER_PATH=checkpoints/voc/${EXP_NAME}/teacher_base/defrcn_det_r101_base${SPLIT_ID}

# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedDistillatedAddingRCNN
# MODEL.ADDITION.TEACHER_TRAINING True
# MODEL.ADDITION.STUDENT_TRAINING False
# MODEL.ADDITION.DISTIL_MODE False
# MODEL.ADDITION.NAME glove
# SOLVER.IMS_PER_BATCH 8
# SOLVER.MAX_ITER 30000
# "
cfg_MODEL="
MODEL.ROI_HEADS.NAME SematicRes5ROIHeads
MODEL.ADDITION.TEACHER_TRAINING True
MODEL.ADDITION.STUDENT_TRAINING False
MODEL.ADDITION.DISTIL_MODE False
MODEL.ADDITION.NAME glove
SOLVER.IMS_PER_BATCH 8
SOLVER.MAX_ITER 30000
"

# SOLVER.BASE_LR 0.01

python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
       OUTPUT_DIR ${TEACHER_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

python3 tools/model_surgery.py --dataset voc --method randinit                                \
   --src-path ${TEACHER_PATH}/model_final.pth                    \
   --save-dir ${TEACHER_PATH}
BASE_WEIGHT=${SAVE_DIR}/teacher_base/defrcn_det_r101_base1/model_reset_surgery.pth
exit
# fine-tuning
# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)

for seed in 1 2 3 4 5 6 7 8 9
do
    for shot in 1 #2 10 #2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        cfg_MODEL="
            MODEL.ROI_HEADS.NAME SematicRes5ROIHeads
            MODEL.ADDITION.TEACHER_TRAINING True
            MODEL.ADDITION.STUDENT_TRAINING False
            MODEL.ADDITION.DISTIL_MODE False
            MODEL.ADDITION.NAME glove
        "


        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${N_GPUS} --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}
        rm ${CONFIG_PATH}
        # rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID} --shot-list 1 # 2 3 5 10  # surmarize all results
