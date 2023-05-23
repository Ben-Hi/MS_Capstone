import os, json, cv2, random, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
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
import settings


def register_dataset(dataset_name, func):
    """Registers a dataset to detectron2 using the provided function.

    Args:
        dataset_name (str): name that the dataset will be registered to detectron2 under.
        func (function): function that will return a list of dicts in detectron2 format. Should be a function from dataset_converters.py

    Returns:
        Metadata: detectron2 metadata object for use in visualizing images in the dataset.
    """
    DatasetCatalog.register(dataset_name, func)
    MetadataCatalog.get(dataset_name).set(thing_classes=settings.THING_CLASSES)
    MetadataCatalog.get(dataset_name).set(thing_colors=settings.THING_COLORS)
    metadata = MetadataCatalog.get(dataset_name)

    return metadata


def save_config(cfg, config_file_name):
    """Saves a detectron2 config node to yaml format in the configs directory.

    Args:
        cfg (CfgNode): detectron2 config node object to be saved
        config_file_name (str): name to save the yaml file under. Must end in '.yaml'
    """
    with open(os.path.join(settings.CONFIG_DIR, config_file_name), "w") as f:
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
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.CLASSES.keys())
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


def vis_random_from_data(data, metadata, num_to_vis=1):
    """Visualizes a number of random images from a dataset.

    Args:
        data (list): detectron2 format dataset where each object in the list holds data for one image.
        metadata (Metadata): detectron2 metadata object for use in visualizing images in the dataset.
        num_to_vis (int, optional): Number of images to visualize. Defaults to 1.
    """
    for d in random.sample(data, num_to_vis):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def save_vis_from_model(predictor, data_dicts, metadata, save_path):
    """Saves the visualizations of a predictor on all images from a dataset.

    Args:
        predictor (): detectron2 model which outputs list[dict] when given an image. See https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format 
        data_dicts (list[dict]): detectron2 standard format list of data where each dict corresponds to one image.
        metadata (Metadata): detectron2 metadata object for use in visualizing images in the dataset.
        save_path (str): absolute path to dir to save images under.
    """
    os.makedirs(save_path, exist_ok=True)
    for d in data_dicts:
        img = cv2.imread(d["file_name"])
        full_save_path = os.path.join(save_path, "pred_" + str(d["image_id"]) + ".png")
        outputs = predictor(img)

        v_pred = Visualizer(img[:, :, ::-1],
                       metadata=metadata, 
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        pred_out = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        img_to_save = pred_out.get_image()[:, :, ::-1]
        cv2.imwrite(full_save_path, img_to_save)


def save_vis_from_data(data_dicts, metadata, save_path):
    """Saves visualizations of all images from a given dataset.

    Args:
        data_dicts (list): detectron2 format dataset where each object in the list holds data for one image.
        metadata (Metadata): detectron2 metadata object for use in visualizing images in the dataset.
        save_path (str): absolute path to dir to save images under.
    """
    for d in data_dicts:
        img = cv2.imread(d["file_name"])
        full_save_path = os.path.join(save_path, "true_" + str(d["image_id"]) + ".png")

        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        img_to_save = out.get_image()[:, :, ::-1]
        cv2.imwrite(full_save_path, img_to_save)


def validate_paths(model_paths=None, config_paths=None):
    """Checks if paths exist and have the appropriate file extensions.

    Args:
        model_paths (List[str], optional): list of model paths to validate. Defaults to None.
        config_paths (List[str], optional): list of config paths to validate. Defaults to None.

    Returns:
        int: 0 if all paths are valid, 1 otherwise
    """
    if model_paths:
        for path in model_paths:
            if not os.path.exists(path):
                print("{} is not a valid path.".format(path))
                return 1

            if not path.endswith('.pth'):
                print("Model weights file must end in .pth")
                return 1

    if config_paths:
        for path in config_paths:
            if not os.path.exists(path):
                print("{} is not a valid path.".format(path))
                return 1

            if not path.endswith('.yaml'):
                print("Config file must end in .yaml")
                return 1

    return 0