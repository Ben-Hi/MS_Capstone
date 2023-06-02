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
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
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
    MetadataCatalog.get(dataset_name).set(evaluator_type=COCOEvaluator)
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
        
        
def load_model(config_path, weights_path):
    """Loads a model from a config file and a model weights file.

    Args:
        config_path (str): path to the config file that the model will be built from. Must end in .yaml
        weights_path (str): absolute path to the weights file for the model. Must end in .pth

    Returns:
        nn.Module, CfgNode: the model and config node associated with the loaded model.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(m2f_settings.CLASSES.keys())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
    cfg.MODEL.WEIGHTS = weights_path
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS

    return model, cfg


def vis_random_model_vs_data(predictor, data_dicts, metadata, num_to_vis=1):
    """Shows a random number of predicted and actual images from data_dicts using the given predictor.

    Args:
        predictor (): detectron2 model which outputs list[dict] when given an image. See https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format 
        data_dicts (list[dict]): detectron2 standard format list of data where each dict corresponds to one image.
        metadata (Metadata): detectron2 Metadata object for visualization.
        num_to_vis (int, optional): Number of images to randomly visualize from data_dicts. Defaults to 1.
    """
    for d in random.sample(data_dicts, num_to_vis):    
        pred_img = cv2.imread(d["file_name"])
        true_img = cv2.imread(d["file_name"])

        outputs = predictor(pred_img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v_pred = Visualizer(pred_img[:, :, ::-1],
                       metadata=metadata, 
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        v_true = Visualizer(true_img[:, :, ::-1],
                       metadata=metadata, 
                       scale=1
        )

        pred_out = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        true_out = v_true.draw_dataset_dict(d)

        cv2.imshow("predicted: " + d["file_name"], pred_out.get_image()[:, :, ::-1])
        cv2.imshow("true: " + d["file_name"], true_out.get_image()[:, :, ::-1])
        cv2.waitKey(0)