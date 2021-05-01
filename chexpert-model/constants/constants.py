"""Define constants to be used throughout the repository."""
from pathlib import Path
import os


# Main directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv('CHEXPERT_DATA_DIR'))
OBJ_EFFICIENCY_BASE_PATH = Path(os.getenv('OBJ_EFFICIENCY_BASE_PATH'))
TEST_SEG_LABELS_GT_BASE_PATH = Path(os.getenv('TEST_SEG_LABELS_GT_BASE_PATH'))
IRNET_MODEL_SAVE_DIR = OBJ_EFFICIENCY_BASE_PATH / "weakly_supervised/saved_models"
TRAIN_CAMS_DATA_DIR = OBJ_EFFICIENCY_BASE_PATH / "train_cams"
TEST_CAMS_DATA_DIR = OBJ_EFFICIENCY_BASE_PATH / "test_cams"
MODEL_NAME = "2ywovex5_epoch=2-chexpert_competition_AUROC=0.89.ckpt"

# Datasets
CHEXPERT = "chexpert"
CUSTOM = "custom"
CXR14 = "cxr14"
POSIX_PATH_PARTS_NUM = 2
POSIX_PATH_PARTS_NUM_TRAIN_CAMS = 3

# Predict config constants
CFG_TASK2MODELS = "task2models"
CFG_AGG_METHOD = "aggregation_method"
CFG_CKPT_PATH = "ckpt_path"
CFG_IS_3CLASS = "is_3class"

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COL_PATH = "Path"
COL_STUDY = "Study"
COL_TASK = "Tasks"
COL_METRIC = "Metrics"
COL_VALUE = "Values"
TASKS = "tasks"
UNCERTAIN = -1
MISSING = -2

# Constants for weakly-supervised segmentation (IRNet)
CHEXPERT_PARENT_TRAIN_CAMS_DIR = TRAIN_CAMS_DATA_DIR / MODEL_NAME
CHEXPERT_PARENT_TEST_CAMS_DIR = TEST_CAMS_DATA_DIR / MODEL_NAME

# CheXpert specific constants
CHEXPERT_DATASET_NAME = "CheXpert-v1.0"
CHEXPERT_PARENT_DATA_DIR = DATA_DIR / "CheXpert"
CHEXPERT_SAVE_DIR = CHEXPERT_PARENT_DATA_DIR / "models/"
CHEXPERT_DATA_DIR = CHEXPERT_PARENT_DATA_DIR / CHEXPERT_DATASET_NAME
CHEXPERT_TEST_DIR = CHEXPERT_PARENT_DATA_DIR / "CodaLab"
CHEXPERT_UNCERTAIN_DIR = CHEXPERT_PARENT_DATA_DIR / "Uncertainty"
CHEXPERT_RAD_PATH = CHEXPERT_PARENT_DATA_DIR / "rad_perf_test.csv"
CHEXPERT_MEAN = [.5020, .5020, .5020]
CHEXPERT_STD = [.085585, .085585, .085585]

CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"
                  ]

LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Support Devices"
                  ]

CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema",
                              "Pleural Effusion"
                              ]

CHEXPERT_SEGMENTATION_CLASSES = ["Enlarged Cardiomediastinum",
                                 "Cardiomegaly",
                                 "Lung Lesion",
                                 "Airspace Opacity",
                                 "Edema",
                                 "Consolidation",
                                 "Atelectasis",
                                 "Pneumothorax",
                                 "Pleural Effusion",
                                 "Support Devices"
                                 ]
SEGMENTATION_COMMON_PATHOLOGIES  = ["Enlarged Cardiomediastinum",
                                    "Support Devices",
                                    "Cardiomegaly",
                                    "Atelectasis"]

CHEXPERT_SEGMENTATION_SAMPLE_CLASSES = ["Enlarged Cardiomediastinum"]

# CXR14 specific constants
CXR14_DATA_DIR = DATA_DIR / CXR14
CXR14_TASKS = ["Cardiomegaly",
               "Emphysema",
               "Pleural Effusion",
               "Hernia",
               "Infiltration",
               "Mass",
               "Nodule",
               "Atelectasis",
               "Pneumothorax",
               "Pleural Thickening",
               "Pneumonia",
               "Fibrosis",
               "Edema",
               "Consolidation"]
CALIBRATION_FILE = "calibration_params.json"

DATASET2TASKS = {CHEXPERT: CHEXPERT_TASKS,
                 CUSTOM: CHEXPERT_TASKS,
                 CXR14: CXR14_TASKS}
EVAL_METRIC2TASKS = {'val_loss': CHEXPERT_TASKS,
                     'cxr14-log_loss': CXR14_TASKS,
                     'chexpert-competition-log_loss': CHEXPERT_COMPETITION_TASKS,
                     'chexpert_competition_AUROC': CHEXPERT_COMPETITION_TASKS}

# CAM specific constants

N_STEPS_IG = 100
INTERNAL_BATCH_SIZE_IG = 10
N_SAMPLES_NT = 5
DEFAULT_STDEV_NT = 0.1
