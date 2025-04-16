from .config import CfgNode as CN

_C = CN()
_C.VERSION = 1
_C.REALTYPE =  "torch.float"
_C.COMPLEXTYPE = "torch.complex64"
_C.VIS_PERIOD = 0
_C.ILT = True
_C.OUTPUT_DIR = "exps/"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.META_SOLVER = "AtomicSolver"
_C.SOLVER.MAX_ITER = 20
_C.SOLVER.SIGMOID_STEEPNESS = 4.0
_C.SOLVER.WEIGHT_EPE = 0.5
_C.SOLVER.WEIGHT_L2 = 1.0
_C.SOLVER.WEIGHT_PVBL2 = 1.0
_C.SOLVER.WEIGHT_PV_Band = 0.
_C.SOLVER.STEP_SIZE = 0.5
_C.SOLVER.LOSS_MIN = 1e12
_C.SOLVER.L2_MIN = 1e12
_C.SOLVER.PVB_MIN = 1e12
_C.SOLVER.LOSS_WEIGHT = None
_C.SOLVER.CORNERS_FORWARD = False

# -----------------------------------------------------------------------------
# Algorithm
# -----------------------------------------------------------------------------
_C.INITIALIZER = CN()
_C.INITIALIZER.TYPE = "levelset" # pixelset, levelset
_C.INITIALIZER.NAME = "LevelSetInitializer" # "LevelSetInitializer" or "PixelInitializer"

# -----------------------------------------------------------------------------
# Algorithm
# -----------------------------------------------------------------------------
_C.ALGORITHM = CN()
_C.ALGORITHM.NAME = "GeneralAlgorithm"
_C.ALGORITHM.OPTIMIZER = "SGD"
_C.ALGORITHM.LR_SCHEDULE = CN()
_C.ALGORITHM.LR_SCHEDULE.OPEN = False
_C.ALGORITHM.LR_SCHEDULE.ITERATION = 30
_C.ALGORITHM.LR_SCHEDULE.STEP_ITERATION = 5
_C.ALGORITHM.LR_SCHEDULE.WEIGHT = 0.5
_C.ALGORITHM.CORNERS = False
_C.ALGORITHM.CORNERS_LIST = []
# support multi-task learning
_C.ALGORITHM.MULTI_TASK_LEARNING = CN()
# save grads
_C.ALGORITHM.MULTI_TASK_LEARNING.SAVE_GRAD = False
_C.ALGORITHM.MULTI_TASK_LEARNING.SAVE_GRAD_ORDER = 2
# select representative grads only
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK = CN()
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.USE_MASK = False
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.MODE = "ratio" # "ratio", "topk"
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.THRESHOLD = 90
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.RATIO = 0.7
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.PRE_AMPLITUDE = True
_C.ALGORITHM.MULTI_TASK_LEARNING.MASK.REFERENCE_INDEX = None
# different MOOs
_C.ALGORITHM.MULTI_TASK_LEARNING.MTL = False
_C.ALGORITHM.MULTI_TASK_LEARNING.SCALER = "MGDA" # grident scalers
_C.ALGORITHM.MULTI_TASK_LEARNING.MODE = 'backward' # autograd
# or 'autograd', but, for ILT task, there is no difference between the both two modes, because the optimized target is shared by different 'objectives'
# mgda
_C.ALGORITHM.MULTI_TASK_LEARNING.MGDA = CN()
_C.ALGORITHM.MULTI_TASK_LEARNING.MGDA.GRAD_NORM = 'none' # ['l2', 'loss', 'loss+', 'none']
# moco
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO = CN()
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO.BETA = 0.5
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO.BETA_SIGMA = 0.5
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO.GAMMA = 0.1
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO.GAMMA_SIGMA = 0.5
_C.ALGORITHM.MULTI_TASK_LEARNING.MOCO.RHO = 0.
# cagrad
_C.ALGORITHM.MULTI_TASK_LEARNING.CAGRAD = CN()
_C.ALGORITHM.MULTI_TASK_LEARNING.CAGRAD.CALPHA = 0.5
_C.ALGORITHM.MULTI_TASK_LEARNING.CAGRAD.RESCALE = 1 # [0, 1, 2]
# gradnorm
_C.ALGORITHM.MULTI_TASK_LEARNING.GRADNORM = CN()
_C.ALGORITHM.MULTI_TASK_LEARNING.GRADNORM.ALPHA = 1.5



# -----------------------------------------------------------------------------
# Lithography
# -----------------------------------------------------------------------------
_C.LITHO_OPERATOR = CN()
_C.LITHO_OPERATOR.NAME = "BasicLithoSim"
_C.LITHO_OPERATOR.REQUIRED = ["KernelDir", "KernelNum",
                              "TargetDensity", "PrintThresh",
                              "PrintSteepness", "DoseMax",
                              "DoseMin", "DoseNom"]
_C.LITHO_OPERATOR.INT_FIELDS = ["KernelNum", ]
_C.LITHO_OPERATOR.FLOAT_FIELDS = ["TargetDensity", "PrintThresh",
                                  "PrintSteepness", "DoseMax",
                                  "DoseMin", "DoseNom"]
_C.LITHO_OPERATOR.DEVICE = "cuda"
_C.LITHO_OPERATOR.TARGET_SHRESHOLD = 0.225
_C.LITHO_OPERATOR.KERNEL_NUM = 24
_C.LITHO_OPERATOR.PRINT_THRESH = 0.5
_C.LITHO_OPERATOR.PRINT_STEEPNESS = 50.0
_C.LITHO_OPERATOR.DOSE_NOM = 1.00
_C.LITHO_OPERATOR.DOSE_MIN = 0.98
_C.LITHO_OPERATOR.DOSE_MAX = 1.02

_C.LITHO_OPERATOR.DOSE_LIST = [0.98, 1.00, 1.02]
_C.LITHO_OPERATOR.PC_MASK_NUM = 0
# filter some corners
_C.LITHO_OPERATOR.MASK = False
_C.LITHO_OPERATOR.MASK_RATIO = 0.8

_C.LITHO_OPERATOR.LOSS = CN()
_C.LITHO_OPERATOR.LOSS.OBJECTIVES = "BasicObjecitvesDict"
_C.LITHO_OPERATOR.LOSS.CURV = None
_C.LITHO_OPERATOR.LOSS.L2_LOSS = "MSE_Loss"
_C.LITHO_OPERATOR.LOSS.PVB_L2_LOSS = "MSE_Loss"
_C.LITHO_OPERATOR.LOSS.PVB_LOSS = "MSE_Loss"
_C.LITHO_OPERATOR.LOSS.EXTRA_MULTI_LOSSES ="{\"loss_name1\":\'L1_Loss\'," \
                                           "\"loss_name2\":\'L1_Loss\',}" # extra objectives for multi task/objective learning
_C.LITHO_OPERATOR.LOSS.EXTRA_MULTI_LOSSES_WEIGHTS = [1.0, 1.0] # with the same length of objectives

_C.LITHO_OPERATOR.KERNEL = CN()
_C.LITHO_OPERATOR.KERNEL.NAME = "Simple_Kernel"
_C.LITHO_OPERATOR.KERNEL.DIRNAME = "./kernel"
_C.LITHO_OPERATOR.KERNEL.TYPES = ["focus", "defocus",
                                  "ct focus", "ct defocus",
                                  "combo focus", "combo defocus",
                                  "combo ct focus", "combo ct defocus"]
                                     # defocus conjuncture combo
_C.LITHO_OPERATOR.KERNEL.PARAMS = "{\"focus\": [False, False, False]," \
                                  "\"defocus\": [True, False, False]," \
                                  "\"ct focus\": [False, True, False]," \
                                  "\"ct defocus\": [True, True,  False]," \
                                  "\"combo focus\": [False, False, True]," \
                                  "\"combo defocus\": [True, False, True]," \
                                  "\"combo ct focus\": [False, True, True]," \
                                  "\"combo ct defocus\": [True, True, True],}"

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------

_C.EVALUATE = CN()
_C.EVALUATE.THRESHOLD = 0.5
_C.EVALUATE.SHOTS = False
_C.EVALUATE.EXPECTED_RESULTS = []

# -----------------------------------------------------------------------------
# Design
# -----------------------------------------------------------------------------
_C.DESIGN = CN()
_C.DESIGN.SCALE = 1
_C.DESIGN.TILE_SIZE_X = 2048
_C.DESIGN.TILE_SIZE_Y = 2048
_C.DESIGN.OFF_SET_X = 512
_C.DESIGN.OFF_SET_Y = 512
_C.DESIGN.ILT_SIZE_X = 1024
_C.DESIGN.ILT_SIZE_Y = 1024


# ---------------------------------------------------------------------------- #
# Trainer
# ---------------------------------------------------------------------------- #
_C.TRAINER = CN()

# Options: WarmupMultiStepLR, WarmupCosineLR.
_C.TRAINER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.TRAINER.MAX_ITER = 40000

_C.TRAINER.BASE_LR = 0.5
# The end lr, only used by WarmupCosineLR
_C.TRAINER.BASE_LR_END = 0.0

_C.TRAINER.MOMENTUM = 0.0

_C.TRAINER.NESTEROV = False

_C.TRAINER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.TRAINER.WEIGHT_DECAY_NORM = 0.0

_C.TRAINER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.TRAINER.STEPS = (30000,)
# Number of decays in WarmupStepWithFixedGammaLR schedule
_C.TRAINER.NUM_DECAYS = 3

_C.TRAINER.WARMUP_FACTOR = 1.0 / 1000
_C.TRAINER.WARMUP_ITERS = 1000
_C.TRAINER.WARMUP_METHOD = "linear"
# Whether to rescale the interval for the learning schedule after warmup
_C.TRAINER.RESCALE_INTERVAL = False

# Save a checkpoint after every this number of iterations
_C.TRAINER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.TRAINER.IMS_PER_BATCH = 1

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.TRAINER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.TRAINER.BIAS_LR_FACTOR = 1.0
_C.TRAINER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY

# Gradient clipping
_C.TRAINER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.TRAINER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.TRAINER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.TRAINER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.TRAINER.AMP = CN({"ENABLED": False})


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ("iccad2013",)


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
# Tf True, when working on datasets that have instance annotations, the
# training dataloader will filter out images without associated annotations
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
