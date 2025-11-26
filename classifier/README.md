# Water Conflict Classifier

SetFit-based multi-label text classifier for identifying water-related conflict events in news headlines.

**Project:** Experimental research supporting the [Pacific Institute's Water Conflict Chronology](https://www.worldwater.org/water-conflict/)  
**Developer:** Baobab Tech  
**License:** [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial)

## Frugal AI: Training with Limited Data

This classifier demonstrates an intentional approach to building AI systems with **limited data** using [SetFit](https://huggingface.co/docs/setfit/en/index) - a framework for few-shot learning with sentence transformers. Rather than defaulting to massive language models (GPT, Claude, or 100B+ parameter models) for simple classification tasks, we fine-tune a small, efficient model (~33M parameters) on a focused dataset.

**Why this matters:** The industry has normalized using trillion-parameter models to classify headlines, answer simple questions, or categorize text - tasks that don't require world knowledge, reasoning, or generative capabilities. This is computationally wasteful and environmentally costly. A properly fine-tuned small model can achieve comparable or better accuracy while using a fraction of the compute resources.

**Our approach:**
- Train on ~600 examples (few-shot learning with SetFit)
- Deploy a 33M parameter model vs. 100B-1T parameter alternatives
- Achieve specialized task performance without the overhead of general-purpose LLMs
- Reduce inference costs and latency by orders of magnitude

This is not about avoiding large models altogether - they're invaluable for complex reasoning tasks. But for targeted classification problems with labeled data, fine-tuning remains the professional, responsible choice.

## Project Structure

Simple, flat structure with shared modules:

```
classifier/
├── __init__.py                         # Package marker
├── data_prep.py                        # Data loading & preprocessing (shared)
├── training_logic.py                   # Core training logic (shared)
├── evaluation.py                       # Model evaluation & metrics (shared)
├── model_card.py                       # Model card generation (shared)
├── train_setfit_headline_classifier.py # Local training script
├── train_on_hf.py                      # HF Jobs training script
├── upload_datasets.py                  # Upload data to HF Hub
├── transform_prep_negatives.py         # Generate negative examples from ACLED
├── classify_headline.py                # Local inference example
├── classify_headline_hub.py            # HF Hub inference example
└── README.md                           # This file
```

### Shared Modules

The training pipeline is split into reusable modules:

- **`data_prep.py`**: Load data from local files or HF Hub, preprocess into multi-label format
- **`training_logic.py`**: Core SetFit training function with configurable parameters
- **`evaluation.py`**: Comprehensive evaluation metrics and reporting
- **`model_card.py`**: Generate model cards with training details and results

Both `train_setfit_headline_classifier.py` (local) and `train_on_hf.py` (cloud) import these shared modules. When HF Jobs runs `train_on_hf.py`, it uploads the entire `classifier/` directory, making all modules available.

---

## Training Options

### Option 1: Local Training

Train on your own hardware with local data files:

```bash
cd classifier
python train_setfit_headline_classifier.py
```

**Pros:** Full control, works offline, no HF account needed  
**Cons:** Requires local GPU (or slow on CPU), manual model management

### Option 2: HF Jobs (Cloud Training)

Train on managed GPUs with automatic model upload to HF Hub:

```bash
cd classifier
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --env HF_ORGANIZATION=your-org \
  --namespace baobabtech \
  --secrets HF_TOKEN \
  train_on_hf.py
```

**Pros:** Fast GPU training (~2-5 min), auto model upload, reproducible  
**Cons:** Requires HF account, data must be on HF Hub

**Learn more:** [Hugging Face Jobs Documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs)

---

## Setup

### For Local Training

1. **Install dependencies:**

```bash
pip install -r ../requirements.txt
```

2. **Prepare training data:**

```bash
# Generate negative examples from ACLED data
cd classifier
python transform_prep_negatives.py

# Place your positives.csv in ../data/
# Format: Headline, Basis (where Basis contains: Trigger, Casualty, Weapon)
```

3. **Train:**

```bash
python train_setfit_headline_classifier.py
```

Model saved to `./water-conflict-classifier/`

---

### For HF Jobs (Cloud Training)

#### Prerequisites

```bash
# Install HF CLI
pip install huggingface-hub[cli]

# Authenticate
hf auth login
```

Get your token from: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### Step 1: Configure HuggingFace Repos

Copy the sample config to create your own:

```bash
cd /path/to/waterconflict
cp config.sample.py config.py
```

Edit `config.py` and set your organization or username:

```python
HF_ORGANIZATION = "my-org-name"  # or "my-username"
```

The `config.py` file is gitignored so your credentials stay local.

#### Step 2: Upload Training Data

```bash
cd classifier
python upload_datasets.py
```

This creates a dataset repository at `YOUR_ORG/water-conflict-training-data` (or `YOUR_USERNAME/...` if using personal account).

#### Step 3: Run Training Job

```bash
cd classifier
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --env HF_ORGANIZATION=baobabtech \
  --secrets HF_TOKEN \
  --namespace baobabtech \
  train_on_hf.py
```

Replace `baobabtech` with your organization name from `config.py`.

**Configuration Options:**

- `--secrets HF_TOKEN`: Authentication (required for private repos/pushing models)
- `--env HF_ORGANIZATION`: Your HF org/username (required - not in git due to .gitignore)
- `--namespace`: Runs job under org account for billing/tracking (optional)
- `--timeout`: Max runtime before auto-termination

**Hardware options:** See [available flavors](https://huggingface.co/docs/trl/main/en/jobs_training#hardware) - recommend `a10g-large` for this task.

---

## Monitoring

```bash
# List jobs
hf jobs ps -a --namespace baobabtech

# Stream logs
hf jobs logs <job_id> --namespace baobabtech

# Cancel job
hf jobs cancel <job_id> --namespace baobabtech
```

---

## Training Pipeline

The script follows the same pipeline as the local version but with HF Hub integration:

1. **Authenticate** with HF Hub (via `HF_TOKEN`)
2. **Load data** from dataset repo (downloads CSVs)
3. **Preprocess** into multi-label format (balances negatives to match positives count)
4. **Split data** (85% train pool / 15% held-out test set)
5. **Sample training data** (600 examples from train pool for efficient few-shot learning)
6. **Train** SetFit model (1 epoch, undersampling strategy)
7. **Evaluate** on held-out test set (F1, accuracy, per-label metrics)
8. **Push to Hub** (model + comprehensive model card with evaluation tables)

Expected runtime: ~2-5 minutes on A10G GPU

---

## After Training

Your model will be at: `https://huggingface.co/YOUR_ORG/water-conflict-classifier` (or `YOUR_USERNAME/...` if using personal account)

Use it with the inference script:

```bash
python classify_headline.py
```

Or directly in Python:

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("YOUR_ORG/water-conflict-classifier")
predictions = model.predict(["Taliban attack dam workers in Afghanistan"])
# Output: [[1, 1, 1]]  # [Trigger, Casualty, Weapon]
```

---

## Troubleshooting

**"Not authenticated"** → Run `hf auth login`

**"Dataset not found"** → Verify `DATASET_REPO` matches uploaded dataset name

**Out of memory** → Reduce `BATCH_SIZE` in script or use smaller GPU flavor

**Job timeout** → Increase `--timeout` value

---

## Local Testing of HF Jobs Script

Test the HF Jobs script locally before submitting:

```bash
cd classifier
pip install uv
uv run train_on_hf.py
```

Note: Still requires dataset on HF Hub and proper authentication.

---

## Configuration Options

### Private Repositories

Set `private=True` in the upload and push methods (check `upload_datasets.py` and `train_on_hf.py`)

### Different Base Model

Edit the BASE_MODEL constant in either training script:

```python
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller/faster
# or
BASE_MODEL = "BAAI/bge-base-en-v1.5"  # Larger/better quality
```

### Additional Secrets

```bash
hf jobs uv run \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
  --env HF_ORGANIZATION=baobabtech \
  --env WANDB_PROJECT=water-conflict \
  train_on_hf.py
```

---

## Data Sources

The training data combines:

- **Positive Examples**: Water conflict headlines from [Pacific Institute Water Conflict Chronology](https://www.worldwater.org/water-conflict/)
- **Negative Examples**: Non-water conflict events from [ACLED](https://acleddata.com/)

Both positive and negative examples are labeled for three categories: Trigger, Casualty, and Weapon.

## Resources

- [HF Jobs Guide](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [UV Script Format](https://docs.astral.sh/uv/guides/scripts/)
- [SetFit Documentation](https://huggingface.co/docs/setfit)
- [Pacific Institute Water Conflict Chronology](https://www.worldwater.org/water-conflict/)
- [ACLED Data](https://acleddata.com/)

---

## License

Copyright © 2025 Baobab Tech

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

You are free to use, share, and adapt this work for non-commercial purposes with appropriate attribution to Baobab Tech. For commercial licensing inquiries, please contact Baobab Tech.
