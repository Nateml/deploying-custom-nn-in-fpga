DETECTORS = ["H1", "L1", "V1"]

# directories
# DATA_DIR = "./../../../../scratch/melissa.lopez/Projects/EarlyWarning/FreqCL_O3/" # CIT cluster
DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data/FreqCL_O3/"  # nikhef cluster
# DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O4" # gaussian O4
TRAIN_DATA_DIR = (
    "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3/"
)
VALID_DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3_validation/"
TEST_DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3_test/"

# parsed data files ({train,validation,test}_folder{k}_batch{n}.npz and
# test_params_*), read by data_prep.into_dataloader and evaluate.py. The float16
# copies (convert_float16.py) hold exactly the same values as the float64 originals
# and are 4x smaller, but only the files in TRAIN_FILES etc. were converted.
# DATA_FILES_DIR = "/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/"  # float64 originals
DATA_FILES_DIR = "/data/gravwav/nmacdona/earlywarningfpgas/parsed_data/new_data_float16/"

# OUTPUT_DIR = "/data/gravwav/amartins/earlywarningfpgas/output/"  # original author
OUTPUT_DIR = "/data/gravwav/nmacdona/earlywarningfpgas/output/"
MODEL_PATH = "/home/anaismartins/Desktop/earlywarningfpgas/output/models/50.0_curriculum_ModifiedPaperNet_3epochs_10samples_8e-05lr_50_minibatch_AdamaxGregoptimizer_0.16decayfactor_128patience_0.01early_stopping_delta_128patience.pt"

# MODEL = "GregNet"
MODEL = "GWaveNet2D"
# MODEL = "GWaveNet2DModified"
FILENAME_ADD = ""  # main.py also adds _{TRAIN_FILES}-{VALIDATION_FILES}-{TEST_FILES}files

# main training parameters
FOLDER = 4
MINIBATCH_SIZE = 50

# data files (of 200 samples) used per folder. The thesis split is 80/10/22
# (16000/2000/4400 samples, thesis section 3.1.2); the data has 100/13/22. The
# original author's code used 40/5/5 (int(10000 * 0.8) etc.), including for the
# distillation teacher.
TRAIN_FILES = 80
VALIDATION_FILES = 13  # all of them; the thesis text says 10 (2000 frames)
TEST_FILES = 22

if MODEL == "GregNet" or MODEL == "GregNet2D" or MODEL == "GregNet2DModified":
    EPOCHS = 6
    TEST_MINIBATCH_SIZE = 50
    LEARNING_RATE = 8e-5
    OPTIMIZER_NAME = "AdamaxGreg"
    EARLY_STOPPING_PATIENCE = 6

elif MODEL == "GWaveNet" or MODEL == "GWaveNet2D" or MODEL == "GWaveNet2DModified":
    EPOCHS = 20
    TEST_MINIBATCH_SIZE = 200
    LEARNING_RATE = 2e-5  # the distillation teacher was trained with 1e-5
    OPTIMIZER_NAME = "AdamW"
    EARLY_STOPPING_PATIENCE = 2


EARLY_STOPPING_MIN_DELTA = 0.0001

# colors
UU_YELLOW = "#FFCD00"
UU_BLACK = "#000000"
UU_CREAM = "#FFE6AB"
UU_ORANGE = "#F3965E"
