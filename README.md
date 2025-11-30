# Water Conflict Research

Experimental research tools to potentially support the [Pacific Institute's Water Conflict Chronology](https://www.worldwater.org/water-conflict/) project.

**Published Package:** [water-conflict-classifier on PyPI](https://pypi.org/project/water-conflict-classifier/)

## Usage

Install setfit and use the trained classifier:

```bash
pip install setfit
```

```python
from setfit import SetFitModel

# Load the trained model from Hugging Face Hub
model = SetFitModel.from_pretrained("baobabtech/water-conflict-classifier")

# Classify headlines
headlines = [
    "Military groups attack workers at the Kajaki Dam in Afghanistan",
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
│   ├── classify.py                   (demo: classify sample headlines)
│   ├── prepare_training_dataset.py   (prepare & version training data)
│   ├── train_on_hf.py                (cloud training with HF Jobs)
│   ├── view_experiments.py           (compare training runs - local)
│   ├── view_evals.py                 (compare training runs - HF Hub)
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
├── experiment_history.jsonl  # Training history (dataset→model mapping)
├── VERSIONING.md             # Dual versioning system docs
└── config.py                 # HF organization config
```

## Quick Start

### 1. Try the Classifier (Demo)

Run the demo script to classify 20 sample headlines with timing metrics:

```bash
python scripts/classify.py
```

This uses the published model from HuggingFace Hub and shows inference performance.

### 2. Full Training Workflow

**Complete guide:** See [scripts/README.md](scripts/README.md#typical-workflow) for detailed step-by-step workflows.

**Quick overview:**
1. **Prepare dataset** - Preprocess, balance, sample, and upload (creates version `d1.0`, `d1.1`, etc.)
2. **Train model** - Cloud (HF Jobs) or local training (creates version `v1.0`, `v1.1`, etc., auto-detects dataset version)
3. **Track results** - Dual versioning links datasets to models for reproducibility
4. **Optimize** - Create 50-500x faster static models (optional)

**Step 1: Prepare training dataset**
```bash
# First time or when data changes (creates d1.0, d1.1, etc.)
python scripts/prepare_training_dataset.py
```

**Step 2: Train model (cloud - recommended):**
```bash
# Auto-detects latest dataset version, creates model version (v1.0, v1.1, etc.)
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=yourorg \
  --namespace yourorg \
  scripts/train_on_hf.py
```

**Or train locally:**
```bash
cd classifier
uv pip install -e .
python train_setfit_headline_classifier.py
```

### 3. Track & Compare Experiments

**Dual versioning system:**
- Dataset versions: `d1.0`, `d1.1`, `d2.0` (from `prepare_training_dataset.py`)
- Model versions: `v1.0`, `v1.1`, `v2.0` (from `train_on_hf.py`)
- Each model tracks which dataset version it used

```bash
# View recent experiments (shows dataset→model mapping)
python scripts/view_experiments.py

# Compare two model versions
python scripts/view_experiments.py --compare v1.0 v1.1

# View from HF Hub
python scripts/view_evals.py
```

See `VERSIONING.md` for full documentation on the dual versioning system.

## Components

### [Classifier Package](classifier/)
Multi-label SetFit classifier for identifying water-related conflict events in news headlines. Classifies into three categories: Trigger, Casualty, Weapon.

**Published to PyPI:** [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/)

**Key Features:**
- Few-shot learning optimized (SetFit)
- Small, efficient models (e.g., BAAI/bge-small-en-v1.5 with ~33M parameters)
- Fast inference (~5-10ms per headline on CPU)
- Published Python package

**This folder contains the package source code.** See `classifier/README.md` and `classifier/PUBLISHING.md`.

### [Scripts](scripts/)
**[→ Full Scripts Documentation](scripts/README.md)**

Utility scripts for the complete ML workflow - from data prep to production deployment:

**Getting Started:**
- 🎯 **Demo** (`classify.py`) - Try the classifier on sample headlines
- 📊 **Data Prep** (`prepare_training_dataset.py`) - Preprocess, balance, and version datasets
- 🚀 **Training** (`train_on_hf.py`) - Train on cloud GPUs with HF Jobs
- 📈 **Analysis** (`view_experiments.py`, `view_evals.py`) - Track and compare experiments
- ⚡ **Optimization** - Create 50-500x faster static models for production

All scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package. See [scripts/README.md](scripts/README.md) for detailed usage and workflows.

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

This project demonstrates intentional "frugal AI" - using small, efficient models (e.g., ~33M parameters) fine-tuned on limited data (~600 examples) instead of defaulting to massive LLMs (100B+ parameters) for simple classification tasks.

**Why this matters:** Properly fine-tuned small models can achieve comparable accuracy to trillion-parameter models for targeted tasks, while using a fraction of compute resources and reducing environmental impact.

## Contact

For questions about this research:

**Olivier Mills**
- Website: [baobabtech.ai](https://baobabtech.ai)
- LinkedIn: [Olivier Mills](https://www.linkedin.com/in/oliviermills/)
