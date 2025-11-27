"""
Hugging Face Configuration

Copy this file to config.py and update with your organization/username.
"""

# Set to your HuggingFace organization name, or your username for personal account
HF_ORGANIZATION = "your-org-name"  # e.g. "acled-data" or "your-username"

# Source dataset repository name (large, unsampled dataset - don't include org/username prefix)
SOURCE_DATASET_REPO_NAME = "water-conflict-source-data"

# Training dataset repository name (sampled data used for training - don't include org/username prefix)
# This uses HF versioning/tags to track different training runs
TRAINING_DATASET_REPO_NAME = "water-conflict-training-data"

# Model repository name (don't include org/username prefix)
MODEL_REPO_NAME = "water-conflict-classifier"

# Evals dataset repository name (don't include org/username prefix)
# Used to track evaluation results across training runs
EVALS_REPO_NAME = "water-conflict-classifier-evals"

