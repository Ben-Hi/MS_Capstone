import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dataset_converters import get_branch_dicts
import settings
import utils


if __name__ == "__main__":
    print("This module demonstrates how to load a model and use it for inference and evaluation.")
    
    weights_path = os.path.join(settings.MODELS_DIR, "overfit/overfit_model_final.pth")
    overfit_model_dir = os.path.join(settings.MODELS_DIR, "overfit")
    config_path = os.path.join(settings.CONFIG_DIR, "branch_mask_rcnn_R_50_FPN_3x_overfit.yaml")
    
    # register the branch dataset
    branch_metadata = utils.register_dataset(settings.WHOLE_DATASET_NAME, get_branch_dicts)
    
    # load the model
    model, cfg = utils.load_model(config_path, weights_path)
    
    # setup inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator(settings.WHOLE_DATASET_NAME, output_dir=overfit_model_dir)
    val_loader = build_detection_test_loader(cfg, settings.WHOLE_DATASET_NAME)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))