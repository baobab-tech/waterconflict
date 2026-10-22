# Water Conflict Classification

Experimental research tools to potentially support the [Pacific Institute's Water Conflict Chronology](https://www.worldwater.org/water-conflict/) project.

> **tl;dr** This tiny model (33m params) classifies events (headlines/descriptions) into water-related conflict events using 3 labels.


**Published Package:** [water-conflict-classifier on PyPI](https://pypi.org/project/water-conflict-classifier/)

## 📈 Performance Notes

The current version v2.5 of the model (as of Dec 1st, 2025) still only achieves the following performance scores at 82% accuracy:

| Overall Metric              | Value  |
|-----------------------------|--------|
| Accuracy (exact match)      | 0.8189 |
| Hamming Loss                | 0.0816 |
| F1 (micro)                  | 0.8656 |
| F1 (macro)                  | 0.8106 |
| F1 (samples)                | 0.7075 |

| Label    | Precision | Recall | F1     | Support |
|----------|-----------|--------|--------|---------|
| Trigger  | 0.8953    | 0.8851 | 0.8902 | 174     |
| Casualty | 0.8889    | 0.9270 | 0.9076 | 233     |
| Weapon   | 0.5493    | 0.7500 | 0.6341 | 52      |

```

```

## 🚀 Quick Start

Try the Classifier (Demo)

Run the demo script to classify 20 sample headlines with timing metrics:

```bash
python scripts/classify.py
```

This uses the published model from HuggingFace Hub and shows inference performance, e.g.,

> You will notice the model shows 3 clear false positives (items 11, 16, 19) where peaceful water developments were misclassified as conflicts. The 0.82 accuracy likely reflects similar edge-case errors.


```
================================================================================
Water Conflict Classifier - Sample Classification Demo
================================================================================
Model: baobabtech/water-conflict-classifier
Headlines to classify: 20
================================================================================

[1/3] Loading model from Hugging Face Hub...
  ✓ Model loaded in 2.41s

[2/3] Running inference...
  ✓ Classified 20 headlines in 0.213s
  ✓ Average time per headline: 10.6ms

[3/3] Results:
================================================================================
 1. 🔴 Militay group attacked workers at the Kajaki Dam construction site in southern Afghanistan, killing three engineers
    → Labels: Casualty

 2. 🔴 Israeli forces bombed water infrastructure in Gaza, leaving thousands without access to clean drinking water
    → Labels: Casualty

 3. 🔴 Armed groups seized control of the Mosul Dam in Iraq during intense fighting between government and insurgent forces
    → Labels: Casualty, Weapon

(...)

11. 🔴 New desalination plant opens in California to address drought conditions with innovative technology
    → Labels: Weapon

12. 🟢 Scientists discover breakthrough water filtration method using graphene-based materials for purification
    → Labels: ❌ No conflict

13. 🟢 City council approves budget for upgrading municipal water treatment systems to meet new standards
    → Labels: ❌ No conflict

(...)

16. 🔴 Tech startup develops smart irrigation system that reduces agricultural water consumption by forty percent
    → Labels: Weapon

(...)

19. 🔴 Community celebrates completion of new well providing clean water access to rural village in Kenya
    → Labels: Trigger

20. 🟢 Weather forecasts predict heavy monsoon rains and potential flooding in South Asian coastal regions
    → Labels: ❌ No conflict

================================================================================
SUMMARY
================================================================================
Total headlines classified: 20
Water conflict detected: 13 (65.0%)
No conflict detected: 7 (35.0%)

Performance:
  - Model load time: 14.51s
  - Total inference time: 0.332s
  - Average per headline: 16.6ms
  - Throughput: 60.2 headlines/second
================================================================================

Label Distribution in Detected Conflicts:
  - Trigger: 4 occurrences
  - Casualty: 6 occurrences
  - Weapon: 5 occurrences
================================================================================
```

## 👩🏽‍💻 Usage

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

## 🗂️ Project Structure

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
│   ├── fetch_serpapi_negatives.py    (real news titles via SerpApi)
│   ├── generate_synthetic_negatives.py (synthetic headlines via Claude Haiku 5.5)
│   ├── verify_negatives.py           (LLM judge: drop anything that is water conflict)
│   ├── build_negatives.py            (dedupe, diversify, hold out → negatives pool)
│   ├── prepare_training_dataset.py   (prepare & version training data)
│   ├── train_on_hf.py                (cloud training with HF Jobs)
│   ├── view_experiments.py           (compare training runs - local)
│   ├── view_evals.py                 (compare training runs - HF Hub)
│   └── See scripts/README.md for details
│
├── data/              # 📂 Training Data
│   ├── positives.csv         (water conflict headlines)
│   ├── real_headlines.csv    (Google News titles)
│   ├── synthetic_headlines.csv (generated headlines)
│   ├── hard_negatives.csv    (peaceful water news)
│   ├── negatives_updated.csv (training-ready negatives pool)
│   └── holdout_negatives.csv (real + synthetic negatives never used in training)
│
├── experiment_history.jsonl  # Training history (dataset→model mapping)
├── VERSIONING.md             # Dual versioning system docs
└── config.py                 # HF organization config
```


### 🏋🏽‍♀️ Full Training Workflow

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

### 🧪 Track & Compare Experiments

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

## 🧱 Components

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

## 📚 Data Sources

**Positive Examples:** Pacific Institute Water Conflict Chronology  
https://www.worldwater.org/water-conflict/

**Negative Examples:** Real news titles from Google News (via [SerpApi](https://serpapi.com/)) and synthetic headlines generated with Claude Haiku 5.5, each checked by an LLM judge against the water conflict definition

**Hard Negatives:** Water-related news that is not conflict (~35% of negatives) to prevent false positives (e.g., utility budgets, flood relief, water research, water sports)

## License

Copyright © 2025 Baobab Tech

Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](http://creativecommons.org/licenses/by-nc/4.0/).

**Non-commercial use only.** 

## 🌱 Frugal AI Philosophy

This project demonstrates intentional "frugal AI" - using small, efficient models (e.g., ~33M parameters) fine-tuned on limited data (~2,000 examples) instead of defaulting to massive LLMs (100B+ parameters) for simple classification tasks.

**Why this matters:** Properly fine-tuned small models can achieve comparable accuracy to trillion-parameter models for targeted tasks, while using a fraction of compute resources and reducing environmental impact.

## Contact

For questions about this research:

**Olivier Mills**
- Website: [baobabtech.ai](https://baobabtech.ai)
- LinkedIn: [Olivier Mills](https://www.linkedin.com/in/oliviermills/)
