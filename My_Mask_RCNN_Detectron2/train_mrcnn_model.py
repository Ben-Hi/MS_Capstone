import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
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
    print("This module trains a pre-trained mask rcnn FPN model on the branch training dataset.")
    if len(sys.argv) != 6:
        print("This module trains a pre-trained mask rcnn model on the branch training dataset.")
        print("Usage: python train_mrcnn_model.py [full_path_to_config] [full_path_to_checkpoint_dir] [num_of_iterations_to_train] [config_save_name] [pretrained_URL]")

    config_path = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    num_train_iters = int(sys.argv[3])
    config_save_name = sys.argv[4]
    weights_url = sys.argv[5]

    if checkpoint_dir == settings.MODELS_DIR:
        print("Cannot use top level models storage as checkpoint dir.")
        exit()

    if utils.validate_paths(None, [config_path]) == 1:
        exit()
        
    output_path = checkpoint_dir
    # register the labelme dataset to detectron2
    train_metadata = utils.register_dataset(settings.TRAIN_DATASET_NAME, get_branch_train_dicts)
    test_metadata = utils.register_dataset(settings.TEST_DATASET_NAME, get_branch_test_dicts)

    # setup the detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = (settings.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (settings.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = settings.NUM_WORKERS
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = weights_url
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = num_train_iters # 72000 on an Nvidia RTX 2060 took approximately 8 hours to complete
    cfg.SOLVER.STEPS = [] # No learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.THING_CLASSES)
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(settings.TEST_DATASET_NAME, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, settings.TEST_DATASET_NAME)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    utils.save_config(cfg, os.path.join(settings.CONFIG_DIR, config_save_name))