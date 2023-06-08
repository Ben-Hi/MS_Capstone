import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask2Former")
import m2f_settings, m2f_utils
from dataset_converters import get_branch_dicts, get_branch_train_dicts, get_branch_test_dicts
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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("This module saves the predictions of a given model on the branch test set.")
        print("Usage:")
        print("\tpython visualize_model.py [full_path_to_config_yaml] [full_path_to_model_weights] [path_to_save_dir]")
        exit()

    config_path = sys.argv[1]
    weights_path = sys.argv[2]
    im_save_dir = sys.argv[3]

    # register the dataset and load the model
    branch_metadata = m2f_utils.register_dataset(m2f_settings.TEST_DATASET_NAME, get_branch_test_dicts)
    _, cfg = m2f_utils.load_model(config_path, weights_path)

    # Inference should use the config with parameters that are used in training
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = m2f_settings.TESTING_THRESHOLD   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    # Load the test data and save the visualizations of the model
    data_dicts = get_branch_test_dicts()
    m2f_utils.save_vis_from_model(predictor, data_dicts, branch_metadata, im_save_dir)