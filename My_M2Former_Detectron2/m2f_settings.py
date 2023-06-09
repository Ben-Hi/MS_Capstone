CLASSES = {"leader": 0, "nonbranch": 1, "other": 2, "sidebranch": 3, "spur": 4}            # mapping between classes and unique id num
THING_CLASSES = ["leader", "nonbranch", "other", "sidebranch", "spur"]                     # list of classes for metadata
THING_COLORS = [(15, 11, 227), (122, 121, 121), (255, 0, 0), (0, 255, 55), (235, 97, 250)] # class colors for visualization
MODELS_DIR = "/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/models"                # dir to create subdirs in to store model.pth files and tensorboard logs
CONFIG_DIR = "/home/hillenb/Desktop/MS_Capstone/My_M2Former_Detectron2/configs"              # dir to store .yaml config files
DATA_DIR = "/home/hillenb/Desktop/MS_Capstone/datasets/original_box_data/UFORozaVideos"    # dir where data is stored
WHOLE_DATASET_NAME = "my_cherry_dataset"                                                   # name of the whole branch dataset
TRAIN_DATASET_NAME = "branch_train"                                                        # name of the training subset
TEST_DATASET_NAME = "branch_test"                                                          # name of the testing subset
DATA_RANGE = ("video_89", "video_184")                                                     # names of the first and last subdirs in the dataset
TRAINING_RANGE = ("video_89", "video_156")                                                 # names of the first and last subdirs in the training dataset
TESTING_RANGE = ("video_157", "video_184")                                                 # names of the first and last subdirs in the testing dataset
NUM_WORKERS = 1                                                                            # 
TESTING_THRESHOLD = 0.7                                                                    # IoU threshold for performance evaluation
IMAGE_SIZE = 640

# urls of all pretrained m2former models. my machine can handle swin-t and smaller.
PRETR_MODEL_URLS = {"r50": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
                   "r101": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl",
                   "swin_t": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl",
                   "swin_s": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl",
                   "swin_b": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl",
                   "swin_b(in_21k)": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl",
                   "swin_l(in_21k)": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl"}