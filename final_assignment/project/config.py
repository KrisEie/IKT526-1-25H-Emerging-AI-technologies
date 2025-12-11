import os

# ==========================================
# Configuration & Hyperparameters
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
GENERATIONS_DIR = os.path.join(OUTPUT_DIR, "generations")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data File
DATA_FILE_NAME = "alpaca_data_cleaned.json"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_TOKEN = os.getenv("HF_TOKEN")  # User provided token (set via env var now)

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training Hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512  # Adjust based on GPU memory

# Data Split Counts
TRAIN_COUNT = 10000
VAL_COUNT = 2000
TEST_COUNT = 2000
