import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from detectron2.engine import DefaultPredictor
from dataset_converters import get_branch_test_dicts
import settings
import utils


def _vis_on_test(config_path, weights_path):
    # register the dataset and load the model
    branch_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)
    _, cfg = utils.load_model(config_path, weights_path)

    # Inference should use the config with parameters that are used in training
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = settings.TESTING_THRESHOLD   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    # Load the ground truth test dataset and visualize against the loaded model
    data_dicts = get_branch_test_dicts()
    utils.vis_random_model_vs_data(predictor, data_dicts, branch_metadata, 3)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("This module compares 3 predicted and actual images for a given model on the test set.")
        print("Usage:")
        print("\tpython visualize_model.py [full_path_to_config_yaml] [full_path_to_model_weights]")
        exit()

    config_path = sys.argv[1]
    weights_path = sys.argv[2]

    if not os.path.exists(config_path):
        print("{} is not a valid path.".format(config_path))
        exit()

    if not os.path.exists(weights_path):
        print("{} is not a valid path.".format(weights_path))
        exit()

    if not config_path.endswith('.yaml'):
        print("Config file must end in .yaml")
        exit()

    if not weights_path.endswith('.pth'):
        print("Model weights file must end in .pth")
        exit()

    _vis_on_test(config_path, weights_path)

    