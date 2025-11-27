# Water Conflict Research

Experimental research tools to potentially support the [Pacific Institute's Water Conflict Chronology](https://www.worldwater.org/water-conflict/) project.

**Published Package:** [water-conflict-classifier on PyPI](https://pypi.org/project/water-conflict-classifier/)

## Usage

Install and use the trained classifier:

```bash
pip install water-conflict-classifier
```

```python
from setfit import SetFitModel

# Load the trained model from Hugging Face Hub
model = SetFitModel.from_pretrained("baobabtech/water-conflict-classifier")

# Classify headlines
headlines = [
    "Taliban attack workers at the Kajaki Dam in Afghanistan",
    "New water treatment plant opens in California"
]

predictions = model.predict(headlines)
# Returns: [[1, 1, 1], [0, 0, 0]]  # [Trigger, Casualty, Weapon]
```

## Project Structure

This is a **mono repo** containing multiple tools for water conflict research:

```
waterconflict/
├── classifier/          # 📦 ML Classifier Package (published to PyPI)
│   ├── Package source code (data_prep, training_logic, evaluation, etc.)
│   ├── Local training script (train_setfit_headline_classifier.py)
│   └── See classifier/README.md for package details
│
├── scripts/            # 🛠️ Utility Scripts (uses published package)
│   ├── classify.py                  (demo: classify sample headlines)
│   ├── transform_prep_negatives.py  (generate negative examples)
│   ├── upload_datasets.py           (upload data to HF Hub)
│   ├── train_on_hf.py               (cloud training with HF Jobs)
│   ├── view_experiments.py          (compare training runs)
│   └── See scripts/README.md for details
│
├── acled/             # 📊 ACLED Data Analysis Tools
│   └── Conflict data analysis and transforms
│
├── data/              # 📂 Training Data
│   ├── positives.csv         (water conflict headlines)
│   ├── negatives.csv         (base ACLED negatives)
│   ├── negatives_updated.csv (training-ready: ACLED + hard negatives)
│   ├── hard_negatives.csv    (peaceful water news)
│   └── ACLED raw data
│
├── experiment_history.jsonl  # Auto-generated training history
├── VERSIONING.md             # Experiment tracking documentation
└── config.py                 # Project configuration
```

## Quick Start

### 1. Try the Classifier (Demo)

Run the demo script to classify 20 sample headlines with timing metrics:

```bash
python scripts/classify.py
```

This uses the published model from HuggingFace Hub and shows inference performance.

### 2. Train Locally

```bash
cd classifier
uv pip install -e .
python train_setfit_headline_classifier.py
```

### 3. Train on HF Jobs (Cloud)

The package is published to PyPI: [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/)

```bash
# From repo root
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=yourorg \
  --namespace yourorg \
  scripts/train_on_hf.py
```

See `scripts/README.md` for cloud training details and `classifier/README.md` for package documentation.

### 4. Track Experiments & Compare Versions

All training runs are automatically versioned and logged:

```bash
# View recent experiments
python scripts/view_experiments.py

# Compare two versions
python scripts/view_experiments.py --compare v1.0 v1.1
```

See `VERSIONING.md` for full documentation on experiment tracking, version management, and HuggingFace Hub integration.

## Components

### [Classifier Package](classifier/)
Multi-label SetFit classifier for identifying water-related conflict events in news headlines. Classifies into three categories: Trigger, Casualty, Weapon.

**Published to PyPI:** [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/)

**Key Features:**
- Few-shot learning optimized (SetFit)
- ~33M parameters (BAAI/bge-small-en-v1.5)
- Fast inference (~5-10ms per headline on CPU)
- Published Python package

**This folder contains the package source code.** See `classifier/README.md` and `classifier/PUBLISHING.md`.

### [Scripts](scripts/)
Utility scripts that use the published package:
- Demo classifier on sample headlines (`classify.py`)
- Generate negative examples from ACLED data (`transform_prep_negatives.py`)
- Generate hard negatives (peaceful water news) to prevent false positives (`generate_hard_negatives.py`)
- Upload datasets to Hugging Face Hub (`upload_datasets.py`)
- Train on HF Jobs cloud infrastructure (`train_on_hf.py`)
- Compare training experiments (`view_experiments.py`)

**See:** `scripts/README.md`

### [ACLED Analysis](acled/)
Tools for analyzing Armed Conflict Location & Event Data (ACLED) to understand conflict patterns and generate training data.

## Data Sources

**Positive Examples:** Pacific Institute Water Conflict Chronology  
https://www.worldwater.org/water-conflict/

**Negative Examples:** Armed Conflict Location & Event Data Project (ACLED) + synthetic hard negatives  
https://acleddata.com/

**Hard Negatives:** Synthetic peaceful water-related news to prevent false positives (e.g., water infrastructure projects, research, conservation initiatives)

## License

Copyright © 2025 Baobab Tech

Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](http://creativecommons.org/licenses/by-nc/4.0/).

**Non-commercial use only.** For commercial licensing inquiries, contact Baobab Tech.

## Frugal AI Philosophy

This project demonstrates intentional "frugal AI" - using a small, efficient model (~33M parameters) fine-tuned on limited data (~600 examples) instead of defaulting to massive LLMs (100B+ parameters) for simple classification tasks.

**Why this matters:** Properly fine-tuned small models can achieve comparable accuracy to trillion-parameter models for targeted tasks, while using a fraction of compute resources and reducing environmental impact.

## Contact

For questions about this research or commercial licensing:
**Baobab Tech**

