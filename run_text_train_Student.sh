# run file: bash bash/danhnt.sh ablations 1
EXP_NAME="singleHeadAtt_Text"
SPLIT_ID=1

# N_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
N_GPUS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
# export CUDA_VISIBLE_DEVICES=4


IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
IMAGENET_PRETRAIN=checkpoints/voc/singleHeadAtt_Text/student_base/defrcn_det_r101_base1/model_final.pth
# IMAGENET_PRETRAIN=checkpoints/voc/singleHeadAtt_Text/student_novel1/1shot_seed1/model_final.pth
# train teacher model
SAVE_DIR=checkpoints/voc/${EXP_NAME}
# TEACHER_PATH=checkpoints/voc/${EXP_NAME}/teacher_base/defrcn_det_r101_base${SPLIT_ID}

# TEACHER_WEIGHT=${TEACHER_PATH}/model_reset_surgery.pth

# # train student model
# cfg_MODEL='
# MODEL.ROI_HEADS.NAME TextRes5ROIHeads
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING True
# MODEL.ROI_HEADS.DISTILLATE True
# MODEL.ROI_HEADS.L2 True
# MODEL.ROI_HEADS.KL True
# '

# STUDENT_PATH=${SAVE_DIR}/student_base/defrcn_det_r101_base${SPLIT_ID}
# TEACHER_NOVEL_DIR=${SAVE_DIR}/teacher_novel${SPLIT_ID}/1shot_seed1
# TEACHER_NOVEL_WEIGHT=${TEACHER_NOVEL_DIR}/model_reset_optimizer.pth


# python3 main.py --num-gpus ${N_GPUS} --eval-only --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
#        OUTPUT_DIR ${STUDENT_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVE_DIR}/student_base/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
    --save-dir ${SAVE_DIR}/student_base/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/student_base/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth

# fine-tuning
# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 #2 10 #2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        cfg_MODEL="
            MODEL.ROI_HEADS.NAME TextRes5ROIHeads
            MODEL.ROI_HEADS.TEACHER_TRAINING False
            MODEL.ROI_HEADS.STUDENT_TRAINING True
            MODEL.ROI_HEADS.DISTILLATE False
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

