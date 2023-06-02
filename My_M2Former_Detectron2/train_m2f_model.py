import os, sys
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/My_Mask_RCNN_Detectron2")
sys.path.append("/home/hillenb/Desktop/MS_Capstone/Mask2Former")
import m2f_settings, m2f_utils
from dataset_converters import get_branch_dicts, get_branch_train_dicts, get_branch_test_dicts
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from mask2former import add_maskformer2_config
from train_net import Trainer


def _process_input():
    """Processes the command line input to train_m2f_model

    Returns:
        list: the url to the model weights, the output directory path, the number of training iterations, and the config file path
    """
    model_key = sys.argv[1]
    num_train_iters = int(sys.argv[2])
    output_dir_name = sys.argv[3]

    if model_key not in m2f_settings.PRETR_MODEL_URLS.keys():
        print("Invalid pre-trained model selection. Use the following command to view options for pre-trained models:")
        print("python train_m2f_model.py")
        exit()
        
    model_url = m2f_settings.PRETR_MODEL_URLS[model_key]
    config_name = model_url.split('/')[-2] + ".yaml"
    config_path = os.path.join(m2f_settings.CONFIG_DIR, config_name)
    output_path = os.path.join(m2f_settings.MODELS_DIR, output_dir_name)
    
    return model_url, output_path, num_train_iters, config_path, config_name


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("This module trains a pre-trained mask2former model on the branch training dataset.")
        print("Usage: python train_m2f_model.py [pretrained_model] [iters_to_train] [output_dir]")
        print("[pretrained_model]: specifies the pretrained model to use. Can be one of: r50 , r101 , swin_t")
        print("[iters_to_train]: the number of iterations to train for. Set to 0 to just perform evaluation. Must be an integer.")
        print("[output_dir]: directory name to store model checkpoints and inference results. Will be created under models/")

    # Shapely outputs warnings about invalid value in intersection because of image size at 640
    try:
        from shapely.errors import ShapelyDeprecationWarning
        import warnings
        warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    except:
        pass
    
    model_url, output_path, num_train_iters, config_path, config_name = _process_input()
    os.makedirs(output_path, exist_ok=True)

    # register the entire branch dataset to detectron2
    train_metadata = m2f_utils.register_dataset(m2f_settings.TRAIN_DATASET_NAME, get_branch_train_dicts)
    test_metadata = m2f_utils.register_dataset(m2f_settings.TEST_DATASET_NAME, get_branch_test_dicts)
    data_dicts = get_branch_dicts()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_url
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    
    # custom config params
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(m2f_settings.THING_CLASSES)
    cfg.DATASETS.TRAIN = (m2f_settings.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (m2f_settings.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = m2f_settings.NUM_WORKERS
    cfg.INPUT.IMAGE_SIZE = m2f_settings.IMAGE_SIZE
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 100 # 72000 on an Nvidia RTX 2060 took approximately 8 hours to complete
    cfg.SOLVER.STEPS = [] # No learning rate decay
    cfg.OUTPUT_DIR = output_path
    
    # verify the output directory and begin training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # setup config file for evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # evaluate the overfit model on the entire dataset
    evaluator = COCOEvaluator(m2f_settings.TEST_DATASET_NAME, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, m2f_settings.TEST_DATASET_NAME)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    m2f_utils.save_config(cfg, os.path.join(m2f_settings.CONFIG_DIR, "branch_" + sys.argv[1] + ".yaml"))
    
    # show a test visualization of the model on one image
    im = cv2.imread("/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/input_branch.png")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    
    v = Visualizer(im[:, :, ::-1], test_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    #cv2.imshow("Test Result", instance_result[:, :, ::-1])
    #cv2.waitKey(0)