import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
import json
import random
import cv2
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import settings

CLASSES = {"leader": 0, "nonbranch": 1, "other": 2, "sidebranch": 3, "spur": 4}
TRAINING_RANGE = ("video_89", "video_156")
TESTING_RANGE = ("video_157", "video_184")
NUM_JSONS = 386

def _getAnnotations(shapes):
    # list[dict]
    annotations = []
    # create a new instance for each annotation
    for shape in shapes:
        new_instance = {}
        new_instance["category_id"] = CLASSES[shape["label"]]

        # bounding boxes in format (x0, y0, w, h)
        new_instance["bbox_mode"] = BoxMode.XYWH_ABS
        x = [x for x, y in shape["points"]]
        x0 = min(x)
        w = max(x) - x0
        y = [y for x, y in shape["points"]]
        y0 = min(y)
        h = max(y) - y0

        new_instance["bbox"] = [x0, y0, w, h]

        new_instance["segmentation"] = []
        
        new_instance["segmentation"].append([p for pair in shape["points"] for p in pair])
        
        annotations.append(new_instance)

    return annotations


def get_branch_test_dicts():
    """Loads the labelme branch test dataset into Detectron2 form.

    Returns:
        List(dict): list of image data where each dictionary corresponds to one image in the test dataset
    """
    DATA_DIR = settings.DATA_DIR
    data_range = TESTING_RANGE
    # sort the subdirectories of videos and get the range of data to load
    subdirs_list = os.listdir(DATA_DIR)
    subdirs_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    subdirs_list = subdirs_list[subdirs_list.index(data_range[0]):]
    subdirs_list = subdirs_list[:subdirs_list.index(data_range[1])+1]

    dataset = []
    id_count = 0
    # create a new dictionary for each json and add it to the dataset
    for subdir in subdirs_list:
        subdir_path = os.path.join(DATA_DIR, subdir)
        subdir_files = os.listdir(subdir_path)
        subdir_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # ignore directories that have no json data to load
        if any("json" in f for f in subdir_files):
            json_names = [f for f in subdir_files if "json" in f]
            json_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            for name in json_names:
                new_data_dict = {}
                img_path = os.path.join(subdir_path, name.replace("json", "png"))
                json_path = os.path.join(subdir_path, name)
                json_file = open(json_path)

                data = json.load(json_file)

                # fill the fields of the new data point
                new_data_dict["file_name"] = img_path
                new_data_dict["height"] = data["imageHeight"]
                new_data_dict["width"] = data["imageWidth"]
                new_data_dict["image_id"] = id_count
                id_count += 1
                new_data_dict["annotations"] = _getAnnotations(data["shapes"])
                dataset.append(new_data_dict)

    return dataset


def get_branch_train_dicts():
    """Loads the labelme branch training dataset into Detectron2 form.

    Returns:
        List(dict): list of image data where each dictionary corresponds to one image in the training dataset
    """
    DATA_DIR = settings.DATA_DIR
    data_range = TRAINING_RANGE
    # sort the subdirectories of videos and get the range of data to load
    subdirs_list = os.listdir(DATA_DIR)
    subdirs_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    subdirs_list = subdirs_list[subdirs_list.index(data_range[0]):]
    subdirs_list = subdirs_list[:subdirs_list.index(data_range[1])+1]

    dataset = []
    id_count = 0
    # create a new dictionary for each json and add it to the dataset
    for subdir in subdirs_list:
        subdir_path = os.path.join(DATA_DIR, subdir)
        subdir_files = os.listdir(subdir_path)
        subdir_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # ignore directories that have no json data to load
        if any("json" in f for f in subdir_files):
            json_names = [f for f in subdir_files if "json" in f]
            json_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            for name in json_names:
                new_data_dict = {}
                img_path = os.path.join(subdir_path, name.replace("json", "png"))
                json_path = os.path.join(subdir_path, name)
                json_file = open(json_path)

                data = json.load(json_file)

                # fill the fields of the new data point
                new_data_dict["file_name"] = img_path
                new_data_dict["height"] = data["imageHeight"]
                new_data_dict["width"] = data["imageWidth"]
                new_data_dict["image_id"] = id_count
                id_count += 1
                new_data_dict["annotations"] = _getAnnotations(data["shapes"])
                dataset.append(new_data_dict)

    return dataset


def get_branch_dicts():
    """Loads the entire labelme branch dataset into Detectron2 form.

    Returns:
        List(dict): list of image data where each dictionary corresponds to one image in the dataset
    """
    DATA_DIR = settings.DATA_DIR
    data_range = (TRAINING_RANGE[0], TESTING_RANGE[1])
    # sort the subdirectories of videos and get the range of data to load
    subdirs_list = os.listdir(DATA_DIR)
    subdirs_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    subdirs_list = subdirs_list[subdirs_list.index(data_range[0]):]
    subdirs_list = subdirs_list[:subdirs_list.index(data_range[1])+1]

    dataset = []
    id_count = 0
    # create a new dictionary for each json and add it to the dataset
    for subdir in subdirs_list:
        subdir_path = os.path.join(DATA_DIR, subdir)
        subdir_files = os.listdir(subdir_path)
        subdir_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # ignore directories that have no json data to load
        if any("json" in f for f in subdir_files):
            json_names = [f for f in subdir_files if "json" in f]
            json_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            for name in json_names:
                new_data_dict = {}
                img_path = os.path.join(subdir_path, name.replace("json", "png"))
                json_path = os.path.join(subdir_path, name)
                json_file = open(json_path)

                data = json.load(json_file)

                # fill the fields of the new data point
                new_data_dict["file_name"] = img_path
                new_data_dict["height"] = data["imageHeight"]
                new_data_dict["width"] = data["imageWidth"]
                new_data_dict["image_id"] = id_count
                id_count += 1
                new_data_dict["annotations"] = _getAnnotations(data["shapes"])
                dataset.append(new_data_dict)

    return dataset


if __name__ == "__main__":
    print("This module implements functions to register cherry branch datasets to detectron2")

    train_dicts = get_branch_train_dicts()
    test_dicts = get_branch_test_dicts()
    all_dicts = get_branch_dicts()