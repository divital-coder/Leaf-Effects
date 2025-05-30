# config.py - RxRx1 Bio-Image Analysis with Batch Effect Correction
import logging
import torch
import numpy as np
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Core Paths ---
# Updated to your actual RxRx1 dataset location
DATASET_BASE_PATH = "/teamspace/studios/this_studio/Comprehensive-RxR1 cellular imaging dataset"

# RxRx1 specific paths - images and metadata are directly in the base path
RXRX1_DATASET_ROOT = os.path.join(DATASET_BASE_PATH, "images")
METADATA_CSV_PATH = os.path.join(DATASET_BASE_PATH, "metadata.csv")

# For development with subset
DEV_MODE = True  # Set to False for full dataset
if DEV_MODE:
    # For development, still use the same images path but limit data in other ways
    # We'll handle subset logic in the dataset class rather than changing paths
    NUM_EPOCHS = 5  # Fewer epochs for testing
    BATCH_SIZE = 16  # Smaller batches for development
else:
    NUM_EPOCHS = 50  # Full training epochs
    BATCH_SIZE = 32  # Full batch size

# Output directories
MODEL_SAVE_PATH = "saved_models"
VISUALIZATION_DIR = "visualizations"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Ensure output directories exist
for dir_path in [MODEL_SAVE_PATH, VISUALIZATION_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- RxRx1 Dataset Constants ---
# RxRx1 has 6 fluorescence channels (w1-w6, not named channels)
NUM_CHANNELS = 6
CHANNEL_NAMES = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']  # Actual RxRx1 channel naming

# Image properties (RxRx1 original: 512x512, we'll resize to 224x224)
ORIGINAL_IMAGE_SIZE = (512, 512)
IMAGE_SIZE_RGB = (224, 224)  # Standard input size for models
IMAGE_SIZE_SPECTRAL = (224, 224)  # Same as RGB for this project

# RxRx1 dataset statistics (updated from metadata schema)
NUM_GENETIC_PERTURBATIONS = 1138  # siRNA classes (corrected from 1108)
NUM_EXPERIMENTAL_PLATES = 51      # Batch sources (varies by experiment)
TOTAL_IMAGES_APPROX = 125510

# Normalization parameters (will be computed from training data)
# Placeholder values - should be updated after computing real statistics
CHANNEL_MEANS = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]


# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPU-optimized settings
if torch.cuda.is_available():
    BATCH_SIZE = 32          # Can handle larger batches
    NUM_WORKERS = 4          # Re-enable multiprocessing
    PIN_MEMORY = True        # Faster CPU->GPU transfer
    NUM_EPOCHS = 50          # Full training
else:
    BATCH_SIZE = 16          # Smaller for CPU
    NUM_WORKERS = 0          # Avoid multiprocessing issues
    PIN_MEMORY = False       # Not useful for CPU
    NUM_EPOCHS = 5           # Quick testing
# Model configurations
MODEL_NAME = "resnet50"  # Feature extractor backbone
PRETRAINED = True

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  # Increased for adversarial training
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4

# Adversarial training specific parameters
LAMBDA_ADVERSARIAL = 1.0      # Weight for adversarial loss
GRADIENT_REVERSAL_ALPHA = 1.0 # Gradient reversal strength
MODEL_TYPE = "dann"           # "dann" for adversarial, "baseline" for standard

# Data Splitting (stratified by both genetic perturbation and plate)
VALIDATION_SPLIT_RATIO = 0.15  # 15% for validation
TEST_SPLIT_RATIO = 0.15        # 15% for test

# Stratification options for batch-aware splitting
STRATIFY_BY_PLATE = True           # Ensure plates are distributed across splits
STRATIFY_BY_PERTURBATION = True    # Ensure genetic perturbations are distributed

# Augmentation & Dataset options
USE_AUGMENTATION = True
AUGMENTATION_STRENGTH = 0.5        # Controls intensity of augmentations
APPLY_PROGRESSION_TRAIN = False    # Not applicable for RxRx1
USE_SPECTRAL_TRAIN = False         # We use all 6 channels as "multi-spectral"

# --- Evaluation Metrics ---
METRICS_AVERAGE = 'macro'  # 'macro', 'micro', or 'weighted'

# Batch effect evaluation parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Batch effect removal metrics thresholds
BATCH_PREDICTION_RANDOM_THRESHOLD = 0.1  # Batch prediction should approach 1/51 ≈ 0.02
MAX_ACCEPTABLE_CV = 0.3                   # Coefficient of Variation across batches

# --- Experiment Tracking ---
EXPERIMENT_NAME = "rxrx1_batch_effects_phase1"
USE_WANDB = False  # Set to True if using Weights & Biases
WANDB_PROJECT = "rxrx1-batch-effects"

# --- Dataset Stage Map (not used for RxRx1, kept for compatibility) ---
DEFAULT_STAGE_MAP = {
    # Not applicable for RxRx1 - genetic perturbations don't have "progression stages"
    # Kept for compatibility with existing code structure
    i: 'perturbation' for i in range(NUM_GENETIC_PERTURBATIONS)
}
