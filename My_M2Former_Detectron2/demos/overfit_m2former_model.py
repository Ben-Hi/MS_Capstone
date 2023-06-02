import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask2Former")
import m2f_settings, m2f_utils
from dataset_converters import get_branch_dicts
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from mask2former import add_maskformer2_config
from train_net import Trainer

DEBUG = True

if __name__ == "__main__":
    print("This module overfits a pretrained Mask2Former net on the entire branch dataset.")
    
    # Shapely outputs warnings about invalid value in intersection because of image size at 640
    try:
        from shapely.errors import ShapelyDeprecationWarning
        import warnings
        warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    except:
        pass
    
    output_dir = os.path.join(m2f_settings.MODELS_DIR, "overfit_testing")

    # register the entire branch dataset to detectron2
    branch_metadata = m2f_utils.register_dataset(m2f_settings.WHOLE_DATASET_NAME, get_branch_dicts)
    data_dicts = get_branch_dicts()
    
    # test if m2former pre-trained is working
    if DEBUG:
        im = cv2.imread("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/input_branch.png")
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/configs/maskformer2_swin_tiny_bs16_50ep.yaml")
        cfg.MODEL.WEIGHTS = m2f_settings.PRETR_MODEL_URLS["swin-t"]
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        
        # custom config params
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
        cfg.DATASETS.TRAIN = (m2f_settings.WHOLE_DATASET_NAME,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = m2f_settings.NUM_WORKERS
        cfg.INPUT.IMAGE_SIZE = m2f_settings.IMAGE_SIZE
        cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = 100 # 72000 on an Nvidia RTX 2060 took approximately 8 hours to complete
        cfg.SOLVER.STEPS = [] # No learning rate decay
        cfg.OUTPUT_DIR = output_dir
        
        # verify the output directory and begin training
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # setup config file for evaluation
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
    
        # evaluate the overfit model on the entire dataset
        evaluator = COCOEvaluator(m2f_settings.WHOLE_DATASET_NAME, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, m2f_settings.WHOLE_DATASET_NAME)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
        
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        
        v = Visualizer(im[:, :, ::-1], branch_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        cv2.imshow("Test Result", instance_result[:, :, ::-1])
        cv2.waitKey(0)

    else:
        # setup the config for Mask2Former with a Resnet 50 backbone
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/maskformer2_R50_bs16_50ep.yaml")
        cfg.MODEL.WEIGHTS = m2f_settings.PRETR_MODEL_URLS["r50"]
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
        cfg.MODEL.RETINANET.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
        cfg.DATASETS.TRAIN = (m2f_settings.WHOLE_DATASET_NAME,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = m2f_settings.NUM_WORKERS
        cfg.INPUT.IMAGE_SIZE = m2f_settings.IMAGE_SIZE
        cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = 1000 # 72000 on an Nvidia RTX 2060 took approximately 8 hours to complete
        cfg.SOLVER.STEPS = [] # No learning rate decay
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default = 512
        cfg.OUTPUT_DIR = output_dir
        m2f_utils.save_config(cfg, "overfit_testing.yaml")
    
        # verify the output directory and begin training
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    
        # setup config file for evaluation
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
    
        # evaluate the overfit model on the entire dataset
        evaluator = COCOEvaluator(m2f_settings.WHOLE_DATASET_NAME, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, m2f_settings.WHOLE_DATASET_NAME)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
        
        #predictor = DefaultPredictor(cfg)
        #
        #d = random.sample(data_dicts, 1)[0]
        #img = cv2.imread(d["file_name"])
        #output = predictor(img)
        #
        #v = Visualizer(img[:, :, ::-1], branch_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        #instance_result = v.draw_instance_predictions(output["instances"].to("cpu")).get_image()
        #cv2.imshow("Instance Result", instance_result[:, :, ::-1])
        #cv2.waitKey(0)