# run file: bash bash/danhnt.sh ablations 1
EXP_NAME="att_roi_heads"
SPLIT_ID=1

# N_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
N_GPUS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
# export CUDA_VISIBLE_DEVICES=4


IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
SAVE_DIR=checkpoints/voc/${EXP_NAME}

# Base distillating
cfg_MODEL="
    MODEL.ROI_HEADS.NAME TextRes5ROIHeads
    MODEL.ADDITION.TEACHER_TRAINING True
    MODEL.ADDITION.STUDENT_TRAINING False
    MODEL.ADDITION.DISTIL_MODE False
    MODEL.ADDITION.NAME glove
    SOLVER.IMS_PER_BATCH 8
    SOLVER.MAX_ITER 30000
"
BASE_DIR=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
python3 main.py --num-gpus ${N_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
       OUTPUT_DIR ${BASE_DIR} ${cfg_MODEL}

exit
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

