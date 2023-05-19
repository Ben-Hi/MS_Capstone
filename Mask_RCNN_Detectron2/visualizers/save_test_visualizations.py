import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from dataset_converters import get_branch_test_dicts
import settings
import utils

if __name__ == "__main__":
    print("This module saves the visualizations of the test segmentation images to data/test_visualizations.")

    save_dir = "/home/hillenb/Desktop/MS_Capstone/data/test_visualizations"

    # register the dataset and load the model
    branch_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)
    
    # Load the test data and save the visualizations of the model
    data_dicts = get_branch_test_dicts()
    utils.save_vis_from_data(data_dicts, branch_metadata, save_dir)