from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.META_ARCHITECTURE = 'SSDDetector'
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.THRESHOLD = 0.5
cfg.MODEL.NUM_CLASSES = 5
# Hard negative mining
cfg.MODEL.NEG_POS_RATIO = 3
cfg.MODEL.CENTER_VARIANCE = 0.1
cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.NAME = 'resnext101'
cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------

cfg.MODEL.PRIORS = CN()
"""
#300x300:
cfg.MODEL.PRIORS.FEATURE_MAPS = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]
cfg.MODEL.PRIORS.STRIDES = [(8,8), (16,16), (32,32), (64,64), (100,100), (300,300)]
cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
"""
"""
# 360x270:
cfg.MODEL.PRIORS.FEATURE_MAPS = [(45, 34), (23, 17), (12, 9), (6,5) , (3,3), (1,1)] # (W, H)
cfg.MODEL.PRIORS.STRIDES = [(8,8), (16,16), (30,30), (60,54), (120,90), (360,270)]
cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
cfg.MODEL.PRIORS.MAX_SIZES = [64, 128, 192, 256, 310, 374]
"""
#512x384:
cfg.MODEL.PRIORS.FEATURE_MAPS = [(64, 48), (32, 24), (16, 12), (8,6) , (4,3), (2,2), (1,1)] # (W, H)
cfg.MODEL.PRIORS.STRIDES = [(8,8), (16,16), (32,32), (64,64), (128,128), (256, 192), (512,384)]
cfg.MODEL.PRIORS.MIN_SIZES = [(20.48, 19.2), (51.2, 38.4), (133.12, 99.84), (215.04, 161.28), (296.96, 222.74), (378.88,284.44), (460.8,345.6)]
cfg.MODEL.PRIORS.MAX_SIZES = [(51.2, 38.4), (133.12, 99.84), (215.04, 161.28), (296.96, 222.72), (378.88, 0.74), (460.8, 345.6), (542.72, 407.04)]

cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2,3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 6, 4, 4]  # number of boxes per feature map location
cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
cfg.MODEL.BOX_HEAD = CN()
cfg.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
cfg.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = (360, 270)	#(W, H)
# Values to be used for image normalization, RGB layout
cfg.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.LR_STEPS = [80000, 100000]
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 2e-2
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 1500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.45
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.MAX_PER_CLASS = -1
cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.EVAL_STEP = 1000 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 2000 # Save checkpoint every save_step
cfg.LOG_STEP = 20 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"