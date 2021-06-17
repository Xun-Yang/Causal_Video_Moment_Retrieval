import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "TAN"
_C.MODEL.LAMBDA = 0.001
_C.MODEL.GAMMA = 1.0
_C.MODEL.SIGMA = 1.0
_C.MODEL.ETA = 1.0
_C.MODEL.DO = 1.0
_C.MODEL.WEIGHT = ""
_C.position_embedding = "sine"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256
_C.INPUT.PRE_QUERY_SIZE = 300

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
_C.DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.TEST_1 = ()
_C.DATASETS.TEST_2 = ()
_C.DATASETS.TEST_3 = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.TCN = CN()
_C.MODEL.TCN.NUM_CLIPS = 128

_C.MODEL.TCN.FEATPOOL = CN()
_C.MODEL.TCN.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.TCN.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.TCN.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.TCN.FEAT2D = CN()
_C.MODEL.TCN.FEAT2D.POOLING_COUNTS = [15,8,8,8]


_C.MODEL.TCN.INTEGRATOR = CN()
_C.MODEL.TCN.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.TCN.INTEGRATOR.LSTM = CN()
_C.MODEL.TCN.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.TCN.INTEGRATOR.LSTM.BIDIRECTIONAL = False

_C.MODEL.TCN.PREDICTOR = CN() 
_C.MODEL.TCN.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.TCN.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.TCN.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.TCN.LOSS = CN()
_C.MODEL.TCN.LOSS.MIN_IOU = 0.3
_C.MODEL.TCN.LOSS.MAX_IOU = 0.7

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.4
_C.TEST.RECALL= "1,5"
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
