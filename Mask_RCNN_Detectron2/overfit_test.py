import os, json, cv2, random, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
import numpy as np
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dataset_converters import get_branch_dicts, get_branch_test_dicts, get_branch_train_dicts
import settings

CLASSES = settings.CLASSES
MY_OUTPUT_DIR = settings.OUTPUT_DIR


def show_random_predictions(data_dicts, predictor, metadata):
    # Pick 3 images and show the true vs predicted instances
    for d in random.sample(data_dicts, 3):
        im_true = cv2.imread(d["file_name"])
        im_pred = cv2.imread(d["file_name"])
        outputs = predictor(im_pred)

        v_pred = Visualizer(im_pred[:, :, ::-1], metadata=metadata, scale = 1, instance_mode=ColorMode.IMAGE_BW)
        v_true = Visualizer(im_true[:, :, ::-1], metadata=metadata, scale = 1, instance_mode=ColorMode.IMAGE_BW)

        pred_out = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        true_out = v_true.draw_dataset_dict(d)

        cv2.imshow("true: " + d["file_name"], true_out.get_image()[:, :, ::-1])
        cv2.imshow("pred: " + d["file_name"], pred_out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":
    print("This module attempts to overfit a pretrained Mask R-CNN model to a single image in the branch dataset")
    DATA_RANGE = ("video_89", "video_184")

    # register the labelme dataset to detectron2
    DatasetCatalog.register("my_cherry_dataset", get_branch_dicts)
    # DatasetCatalog.register("branch_train", get_branch_train_dicts)
    # DatasetCatalog.register("branch_test", get_branch_test_dicts)

    MetadataCatalog.get("my_cherry_dataset").set(thing_classes=["leader", "nonbranch", "other", "sidebranch", "spur"])
    # MetadataCatalog.get("branch_train").set(thing_classes=["leader", "nonbranch", "other", "sidebranch", "spur"])
    # MetadataCatalog.get("branch_test").set(thing_classes=["leader", "nonbranch", "other", "sidebranch", "spur"])

    branch_metadata = MetadataCatalog.get("my_cherry_dataset")
    # train_metadata = MetadataCatalog.get("branch_train")
    # test_metadata = MetadataCatalog.get("branch_test")

    # setup the detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_cherry_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 80000 # 15000 on an Nvidia RTX 2060 took approximately 1.5 hours to complete
    cfg.SOLVER.STEPS = [] # No learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES.keys())
    cfg.OUTPUT_DIR = MY_OUTPUT_DIR

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    #data_dicts = get_branch_dicts(DATA_RANGE)
    #show_random_predictions(data_dicts, predictor, branch_metadata)

    evaluator = COCOEvaluator("my_cherry_dataset", output_dir=MY_OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_cherry_dataset")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))