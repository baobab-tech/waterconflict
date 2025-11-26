#!/usr/bin/env python
"""
Classify headlines using the trained water conflict classifier from HF Hub.

Usage:
    python classify_headline.py
"""

import sys
from pathlib import Path
from setfit import SetFitModel

# Add project root to path and load configuration
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import HF_ORGANIZATION, MODEL_REPO_NAME
except ImportError:
    print("Error: config.py not found. Copy config.sample.py to config.py and set your HF_ORGANIZATION")
    sys.exit(1)

# Load model from HF Hub
MODEL_REPO = f"{HF_ORGANIZATION}/{MODEL_REPO_NAME}"
print(f"Loading model from: {MODEL_REPO}")
model = SetFitModel.from_pretrained(MODEL_REPO)
print("✓ Model loaded\n")

# Classify new headlines
headlines = [
    "Dam workers killed in attack",
    "New irrigation system installed",
    "Taliban attack workers at the Kajaki Dam in Afghanistan",
    "Water treatment plant opens in California"
]

predictions = model.predict(headlines)

# Display results
LABEL_NAMES = ['Trigger', 'Casualty', 'Weapon']
print("Results:")
print("=" * 80)
for text, pred in zip(headlines, predictions):  # type: ignore
    labels_detected = [LABEL_NAMES[i] for i, val in enumerate(pred) if val == 1]
    label_str = ", ".join(labels_detected) if labels_detected else "None"
    print(f"\nHeadline: {text}")
    print(f"Labels:   {label_str}")
print("\n" + "=" * 80)

