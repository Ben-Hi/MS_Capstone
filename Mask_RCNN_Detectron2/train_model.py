import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dataset_converters import get_branch_test_dicts, get_branch_train_dicts
import settings, utils


if __name__ == "__main__":
    print("This module trains a pre-trained mask rcnn FPN model on the branch training dataset.")

    output_path = os.path.join(settings.MODELS_DIR, "pretrained_fpn_50")
    # register the labelme dataset to detectron2
    train_metadata = utils.register_dataset(settings.TRAIN_DATASET_NAME, get_branch_train_dicts)
    test_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)

    # setup the detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (settings.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (settings.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = settings.NUM_WORKERS
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 80000 # 72000 on an Nvidia RTX 2060 took approximately 8 hours to complete
    cfg.SOLVER.STEPS = [] # No learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.THING_CLASSES)
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(settings.TEST_DATASET_NAME, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, settings.TEST_DATASET_NAME)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    utils.save_config(cfg, os.path.join(settings.CONFIG_DIR, "branch_fpn_50_3x.yaml"))