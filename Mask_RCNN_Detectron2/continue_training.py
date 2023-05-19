import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask_RCNN_Detectron2")
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dataset_converters import get_branch_test_dicts, get_branch_train_dicts
import settings, utils


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("This module continues to train a given model.")
        print("Usage: python continue_training.py [full_path_to_config] [full_path_to_checkpoint_dir] [num_of_iterations_to_train] [config_save_name]")

    config_path = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    additional_iters = int(sys.argv[3])
    config_save_name = sys.argv[4]

    if checkpoint_dir == settings.MODELS_DIR:
        print("Cannot use top level models storage as checkpoint dir.")
        exit()

    if utils.validate_paths(None, [config_path]) == 1:
        exit()

    # register the branch dataset to detectron2
    train_metadata = utils.register_dataset(settings.TRAIN_DATASET_NAME, get_branch_train_dicts)
    test_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)

    # setup the detectron2 config
    _, cfg = utils.load_model(config_path, os.path.join(checkpoint_dir, "model_final.pth"))
    cfg.OUTPUT_DIR = checkpoint_dir
    cfg.SOLVER.MAX_ITER += additional_iters

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(settings.TEST_DATASET_NAME, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, settings.TEST_DATASET_NAME)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    print(cfg.SOLVER.BASE_LR)
    utils.save_config(cfg, os.path.join(settings.CONFIG_DIR, config_save_name))