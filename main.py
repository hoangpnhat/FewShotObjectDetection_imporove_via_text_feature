import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup
import torch


def align_iter(cfg, cur_gpu):
    # alpha = 8//cur_gpu
    # alpha = alpha//2
    alpha = 1
    cfg.defrost()
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/alpha
    cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH/alpha

    # print('before:', cfg.SOLVER.STEPS)
    # alpha *= 2
    # alpha /= 2
    # print(alpha)
    cfg.SOLVER.STEPS = tuple([int(i*alpha) for i in cfg.SOLVER.STEPS])
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER*alpha)
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.WARMUP_ITERS*alpha)
    # cfg.TEST.EVAL_PERIOD = 100000000000
    # print(cfg.SOLVER.MAX_ITER)
    # assert 0


def align_iter_student(cfg):
    alpha = 1
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/alpha
    # alpha = 2
    cfg.SOLVER.STEPS = tuple([int(i/alpha) for i in cfg.SOLVER.STEPS])
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER/alpha)
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.WARMUP_ITERS/alpha)

    print('student config')
    print('cfg.SOLVER.BASE_LR:', cfg.SOLVER.BASE_LR)
    print('cfg.SOLVER.STEPS:', cfg.SOLVER.STEPS)
    print('cfg.SOLVER.MAX_ITER:', cfg.SOLVER.MAX_ITER)
    print('cfg.SOLVER.WARMUP_ITERS:', cfg.SOLVER.WARMUP_ITERS)
    print('cfg.TEST.EVAL_PERIOD:', cfg.TEST.EVAL_PERIOD)


def add_new_config(cfg):
    # define new attribute
    cfg.MODEL.META_LOSS_WEIGHT = 0.0
    cfg.MODEL.SYN_LOSS_WEIGHT = 1e-1
    cfg.MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR = False
    cfg.MODEL.OT_AAT = False
    cfg.MODEL.ROI_HEADS.FREEZE_MPL = False
    cfg.MODEL.ROI_HEADS.FREEZE_AFFINE_LAYERS = False
    cfg.MODEL.ROI_HEADS.USE_MEMORY = False
    cfg.MODEL.ROI_HEADS.USE_OT = False
    cfg.MODEL.ROI_HEADS.REPEATED_TIME = 16
    cfg.MODEL.ROI_HEADS.MEM_CAPACITY = 128

    cfg.MODEL.ROI_HEADS.GEN_L2 = False
    cfg.MODEL.ROI_HEADS.GEN_KL = False
    cfg.MODEL.ROI_HEADS.GEN_OT = True

    cfg.MODEL.ROI_HEADS.FACTORS = tuple([])
    cfg.MODEL.ROI_HEADS.USE_BBX = True
    cfg.MODEL.ROI_HEADS.FREE_ATTENTION = False
    cfg.MODEL.ROI_HEADS.TEACHER_TRAINING = False
    cfg.MODEL.ROI_HEADS.STUDENT_TRAINING = False
    cfg.MODEL.ROI_HEADS.DISTILLATE = False
    cfg.MODEL.ROI_HEADS.DROPOUT_ATTENTION = 0.0
    cfg.MODEL.ROI_HEADS.SUPER_TRAINING = True

    cfg.MODEL.ROI_HEADS.L2 = True
    cfg.MODEL.ROI_HEADS.L2_COSINE = False
    cfg.MODEL.ROI_HEADS.KL = True
    cfg.MODEL.ROI_HEADS.KL_TEMP = 1
    cfg.MODEL.NUM_CLUSTER = 16
    pseudo_class = {
        'aeroplane': [7, 4, 5, 1, 0, 7, 2, 7, 5, 4, 6, 4, 2],
        'bicycle': [7, 1, 4, 4, 0, 1, 4, 4, 4, 1, 4, 3, 3],
        'boat': [3, 4, 0, 5, 0, 6, 5, 3, 5, 2, 0, 2, 0],
        'bottle': [6, 2, 6, 3, 4, 4, 2, 5, 7, 6, 6, 4, 4],
        'car': [3, 1, 3, 4, 0, 2, 5, 5, 4, 5, 4, 3, 3],
        'cat': [4, 3, 6, 5, 6, 5, 6, 6, 0, 5, 2, 1, 1],
        'chair': [3, 7, 7, 5, 2, 6, 1, 2, 7, 6, 7, 4, 5],
        'diningtable': [6, 1, 6, 7, 4, 1, 1, 2, 7, 3, 7, 0, 7],
        'dog': [0, 3, 6, 6, 2, 4, 6, 0, 0, 0, 5, 1, 1],
        'horse': [0, 1, 6, 5, 4, 3, 0, 4, 1, 5, 5, 2, 1],
        'person': [0, 3, 6, 3, 1, 1, 7, 7, 3, 0, 3, 4, 3],
        'pottedplant': [2, 6, 4, 3, 5, 5, 5, 6, 6, 3, 1, 5, 4],
        'sheep': [5, 6, 2, 0, 7, 5, 3, 0, 1, 5, 5, 1, 1],
        'train': [1, 0, 1, 2, 0, 3, 7, 5, 4, 2, 4, 7, 3],
        'tvmonitor': [3, 3, 7, 0, 1, 4, 3, 3, 2, 5, 4, 6, 0],
        'bird': [2, 1, 6, 0, 3, 5, 4, 6, 0, 7, 3, 0, 6],
        'bus': [1, 7, 1, 2, 0, 0, 7, 5, 7, 2, 4, 7, 3],
        'cow': [5, 5, 5, 0, 7, 5, 4, 1, 1, 5, 6, 5, 6],
        'motorbike': [7, 2, 3, 4, 0, 1, 5, 7, 4, 1, 4, 3, 3],
        'sofa': [6, 2, 3, 7, 4, 4, 1, 2, 7, 6, 7, 7, 5],
    }
    cfg.MODEL.K_CLASS = 8
    cfg.MODEL.ROI_HEADS.PSEUDO_CLASS_VOC = [
        [key, val] for key, val in pseudo_class.items()]
    cfg.MODEL.ROI_HEADS.NUM_GROUP_SUPER = 13

    cfg.SUPER_CLASS = [{'supercategory': 'person',
                        'id': 1,
                        'name': 'person',
                        'super_id': 0,
                        'voc_name': 'person'},
                       {'supercategory': 'vehicle',
                        'id': 2,
                        'name': 'bicycle',
                        'super_id': 1,
                        'voc_name': 'bicycle'},
                       {'supercategory': 'vehicle',
                        'id': 3,
                        'name': 'car',
                        'super_id': 1,
                        'voc_name': 'car'},
                       {'supercategory': 'vehicle',
                        'id': 4,
                        'name': 'motorcycle',
                        'super_id': 1,
                        'voc_name': 'motorbike'},
                       {'supercategory': 'vehicle',
                        'id': 5,
                        'name': 'airplane',
                        'super_id': 1,
                        'voc_name': 'aeroplane'},
                       {'supercategory': 'vehicle',
                        'id': 6,
                        'name': 'bus',
                        'super_id': 1,
                        'voc_name': 'bus'},
                       {'supercategory': 'vehicle',
                        'id': 7,
                        'name': 'train',
                        'super_id': 1,
                        'voc_name': 'train'},
                       {'supercategory': 'vehicle',
                        'id': 8,
                        'name': 'truck',
                        'super_id': 1,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'vehicle',
                        'id': 9,
                        'name': 'boat',
                        'super_id': 1,
                        'voc_name': 'boat'},
                       {'supercategory': 'outdoor',
                        'id': 10,
                        'name': 'traffic light',
                        'super_id': 2,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'outdoor',
                        'id': 11,
                        'name': 'fire hydrant',
                        'super_id': 2,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'outdoor',
                        'id': 13,
                        'name': 'stop sign',
                        'super_id': 2,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'outdoor',
                        'id': 14,
                        'name': 'parking meter',
                        'super_id': 2,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'outdoor',
                        'id': 15,
                        'name': 'bench',
                        'super_id': 2,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'animal',
                        'id': 16,
                        'name': 'bird',
                        'super_id': 3,
                        'voc_name': 'bird'},
                       {'supercategory': 'animal',
                        'id': 17,
                        'name': 'cat',
                        'super_id': 3,
                        'voc_name': 'cat'},
                       {'supercategory': 'animal',
                        'id': 18,
                        'name': 'dog',
                        'super_id': 3,
                        'voc_name': 'dog'},
                       {'supercategory': 'animal',
                        'id': 19,
                        'name': 'horse',
                        'super_id': 3,
                        'voc_name': 'horse'},
                       {'supercategory': 'animal',
                        'id': 20,
                        'name': 'sheep',
                        'super_id': 3,
                        'voc_name': 'sheep'},
                       {'supercategory': 'animal',
                        'id': 21,
                        'name': 'cow',
                        'super_id': 3,
                        'voc_name': 'cow'},
                       {'supercategory': 'animal',
                        'id': 22,
                        'name': 'elephant',
                        'super_id': 3,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'animal',
                        'id': 23,
                        'name': 'bear',
                        'super_id': 3,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'animal',
                        'id': 24,
                        'name': 'zebra',
                        'super_id': 3,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'animal',
                        'id': 25,
                        'name': 'giraffe',
                        'super_id': 3,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'accessory',
                        'id': 27,
                        'name': 'backpack',
                        'super_id': 4,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'accessory',
                        'id': 28,
                        'name': 'umbrella',
                        'super_id': 4,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'accessory',
                        'id': 31,
                        'name': 'handbag',
                        'super_id': 4,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'accessory',
                        'id': 32,
                        'name': 'tie',
                        'super_id': 4,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'accessory',
                        'id': 33,
                        'name': 'suitcase',
                        'super_id': 4,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 34,
                        'name': 'frisbee',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 35,
                        'name': 'skis',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 36,
                        'name': 'snowboard',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 37,
                        'name': 'sports ball',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 38,
                        'name': 'kite',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 39,
                        'name': 'baseball bat',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 40,
                        'name': 'baseball glove',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 41,
                        'name': 'skateboard',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 42,
                        'name': 'surfboard',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'sports',
                        'id': 43,
                        'name': 'tennis racket',
                        'super_id': 5,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 44,
                        'name': 'bottle',
                        'super_id': 6,
                        'voc_name': 'bottle'},
                       {'supercategory': 'kitchen',
                        'id': 46,
                        'name': 'wine glass',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 47,
                        'name': 'cup',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 48,
                        'name': 'fork',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 49,
                        'name': 'knife',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 50,
                        'name': 'spoon',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'kitchen',
                        'id': 51,
                        'name': 'bowl',
                        'super_id': 6,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 52,
                        'name': 'banana',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 53,
                        'name': 'apple',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 54,
                        'name': 'sandwich',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 55,
                        'name': 'orange',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 56,
                        'name': 'broccoli',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 57,
                        'name': 'carrot',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 58,
                        'name': 'hot dog',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 59,
                        'name': 'pizza',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 60,
                        'name': 'donut',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'food',
                        'id': 61,
                        'name': 'cake',
                        'super_id': 7,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'furniture',
                        'id': 62,
                        'name': 'chair',
                        'super_id': 8,
                        'voc_name': 'chair'},
                       {'supercategory': 'furniture',
                        'id': 63,
                        'name': 'couch',
                        'super_id': 8,
                        'voc_name': 'sofa'},
                       {'supercategory': 'furniture',
                        'id': 64,
                        'name': 'potted plant',
                        'super_id': 8,
                        'voc_name': 'pottedplant'},
                       {'supercategory': 'furniture',
                        'id': 65,
                        'name': 'bed',
                        'super_id': 8,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'furniture',
                        'id': 67,
                        'name': 'dining table',
                        'super_id': 8,
                        'voc_name': 'diningtable'},
                       {'supercategory': 'furniture',
                        'id': 70,
                        'name': 'toilet',
                        'super_id': 8,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'electronic',
                        'id': 72,
                        'name': 'tv',
                        'super_id': 9,
                        'voc_name': 'tvmonitor'},
                       {'supercategory': 'electronic',
                        'id': 73,
                        'name': 'laptop',
                        'super_id': 9,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'electronic',
                        'id': 74,
                        'name': 'mouse',
                        'super_id': 9,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'electronic',
                        'id': 75,
                        'name': 'remote',
                        'super_id': 9,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'electronic',
                        'id': 76,
                        'name': 'keyboard',
                        'super_id': 9,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'electronic',
                        'id': 77,
                        'name': 'cell phone',
                        'super_id': 9,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'appliance',
                        'id': 78,
                        'name': 'microwave',
                        'super_id': 10,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'appliance',
                        'id': 79,
                        'name': 'oven',
                        'super_id': 10,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'appliance',
                        'id': 80,
                        'name': 'toaster',
                        'super_id': 10,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'appliance',
                        'id': 81,
                        'name': 'sink',
                        'super_id': 10,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'appliance',
                        'id': 82,
                        'name': 'refrigerator',
                        'super_id': 10,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 84,
                        'name': 'book',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 85,
                        'name': 'clock',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 86,
                        'name': 'vase',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 87,
                        'name': 'scissors',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 88,
                        'name': 'teddy bear',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 89,
                        'name': 'hair drier',
                        'super_id': 11,
                        'voc_name': 'NoVOC'},
                       {'supercategory': 'indoor',
                        'id': 90,
                        'name': 'toothbrush',
                        'super_id': 11,
                        'voc_name': 'NoVOC'}]


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(
                dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name,output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_new_config(cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # align_iter(cfg, args.num_gpus)
    if cfg.MODEL.ROI_HEADS.STUDENT_TRAINING:
        align_iter_student(cfg)
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER//3 + \
        100 if cfg.TEST.EVAL_PERIOD == 1000 else cfg.TEST.EVAL_PERIOD
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # if args.init_only:
    #     cfg.SOLVER.MAX_ITER = 1  # trick to save model
        # trainer = Trainer(cfg)
        # trainer.resume_or_load(resume=args.resume)
        # trainer.train()

        # model = Trainer.build_model(cfg)
        # DetectionCheckpointer(model).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )

        # save_name = 'init_weights.pth'
        # save_path = os.path.join(cfg.OUTPUT_DIR, save_name)

        # torch.save(model, save_path)
        # print('save changed ckpt to {}'.format(save_path))
        # return trainer.train()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
