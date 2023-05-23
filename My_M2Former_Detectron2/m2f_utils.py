import os, json, cv2, random, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask2Former")
import numpy as np
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dataset_converters import get_branch_dicts, get_branch_test_dicts, get_branch_train_dicts
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import m2f_settings


def register_dataset(dataset_name, func):
    """Registers a dataset to detectron2 using the provided function.

    Args:
        dataset_name (str): name that the dataset will be registered to detectron2 under.
        func (function): function that will return a list of dicts in detectron2 format. Should be a function from dataset_converters.py

    Returns:
        Metadata: detectron2 metadata object for use in visualizing images in the dataset.
    """
    DatasetCatalog.register(dataset_name, func)
    MetadataCatalog.get(dataset_name).set(thing_classes=m2f_settings.THING_CLASSES)
    MetadataCatalog.get(dataset_name).set(thing_colors=m2f_settings.THING_COLORS)
    metadata = MetadataCatalog.get(dataset_name)

    return metadata


def save_config(cfg, config_file_name):
    """Saves a detectron2 config node to yaml format in the configs directory.

    Args:
        cfg (CfgNode): detectron2 config node object to be saved
        config_file_name (str): name to save the yaml file under. Must end in '.yaml'
    """
    with open(os.path.join(m2f_settings.CONFIG_DIR, config_file_name), "w") as f:
        f.write(cfg.dump())   # save config to file