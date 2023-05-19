import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from dataset_converters import get_branch_test_dicts
from detectron2.engine import DefaultPredictor
import settings, utils


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("This module saves the predictions of a given model on the branch test set.")
        print("Usage:")
        print("\tpython visualize_model.py [full_path_to_config_yaml] [full_path_to_model_weights] [path_to_save_dir]")
        exit()

    config_path = sys.argv[1]
    weights_path = sys.argv[2]
    im_save_dir = sys.argv[3]

    if utils.validate_paths([weights_path], [config_path]) == 1:
        exit()

    # register the dataset and load the model
    branch_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)
    _, cfg = utils.load_model(config_path, weights_path)

    # Inference should use the config with parameters that are used in training
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = settings.TESTING_THRESHOLD   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    # Load the test data and save the visualizations of the model
    data_dicts = get_branch_test_dicts()
    utils.save_vis_from_model(predictor, data_dicts, branch_metadata, im_save_dir)