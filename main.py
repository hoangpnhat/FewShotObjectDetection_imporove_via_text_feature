import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from detectron2.config import CfgNode as CN
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup
import torch



def batch_size_based_cfg_adjustment(cfg):
    alpha = 16 // cfg.SOLVER.IMS_PER_BATCH
    # alpha = 1
    cfg.defrost()
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/alpha
    cfg.SOLVER.STEPS = tuple([int(step*alpha) for step in cfg.SOLVER.STEPS])
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER*alpha)
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

def add_new_configs(cfg):
    cfg.MODEL.RPN.ADDITION_MODEL = "glove"
    cfg.MODEL.RPN.ADDITION = False
    cfg.MODEL.DISTILLATION = CN()
    cfg.MODEL.DISTILLATION.TEACHER_TRAINING = False
    cfg.MODEL.DISTILLATION.STUDENT_TRAINING = False
    cfg.ADDITION = CN()
    cfg.ADDITION.SEMANTIC_DIM = 300


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
    add_new_configs(cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # align_iter(cfg, args.num_gpus)
    # if cfg.MODEL.ROI_HEADS.STUDENT_TRAINING:
    #     align_iter_student(cfg)
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
