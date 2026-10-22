# Utility Scripts

Scripts for data preparation, dataset management, cloud training, and experiment tracking. These scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package.

## Quick Reference

| Category | Script | Purpose |
|----------|--------|---------|
| **🎯 Demo** | `classify.py` | Quick demo of classifier on sample headlines |
| **📊 Data Prep** | `fetch_serpapi_negatives.py` | Real news titles via SerpApi |
| | `generate_synthetic_negatives.py` | Synthetic headlines via Claude Haiku 5.5 |
| | `generate_hard_negatives.py` | Create peaceful water news (hard negatives) |
| | `verify_negatives.py` | LLM judge: drop anything that is water conflict |
| | `build_negatives.py` | Dedupe, diversify, hold out → negatives pool |
| | `upload_datasets.py` | Upload source data to HF Hub |
| | `prepare_training_dataset.py` | Create training-ready dataset |
| **🚀 Training** | `train_on_hf.py` | Train on HF Jobs (cloud GPUs) |
| | `train_on_hf_optuna.py` | Train with Optuna hyperparameter search |
| **⚡ Optimization** | `distill_to_static.py` | Create 50-500x faster static model |
| | `inference_static.py` | Test the static model |
| **📈 Analysis** | `view_experiments.py` | Compare training runs locally |
| | `view_evals.py` | Compare results from HF Hub |
| | `demo_versioning.py` | Demo versioning system |

**New to this project?** Start with:
1. `classify.py` - See the classifier in action
2. **[Typical Workflow](#typical-workflow)** (bottom of this doc) - Step-by-step guide

---

## Script Details

### `view_experiments.py`
View and compare model training experiments from local experiment history. All training runs are automatically logged with metrics and configs.

**Usage:**
```bash
# View recent experiments
python scripts/view_experiments.py

# View all experiments
python scripts/view_experiments.py --all

# Compare two versions
python scripts/view_experiments.py --compare v1.0 v1.1

# Compare using different metric
python scripts/view_experiments.py --compare v1.0 v1.2 --metric accuracy
```

**What it does:**
- Displays experiment history from `experiment_history.jsonl`
- Shows training configs, metrics, and timestamps
- Compares performance between versions
- Helps identify improvements and regressions

**See:** `../VERSIONING.md` for full documentation

---

### `view_evals.py`
View and compare evaluation results from HuggingFace evals dataset. Training runs automatically upload results to HF for easy comparison across experiments.

**Usage:**
```bash
# View all experiments sorted by F1 score
python scripts/view_evals.py

# Show top 10 experiments
python scripts/view_evals.py --top 10

# Compare two specific versions
python scripts/view_evals.py --compare v1.0 v1.1

# Analyze impact of hyperparameters
python scripts/view_evals.py --analyze sample_size
python scripts/view_evals.py --analyze num_epochs

# Show detailed per-label metrics for top experiments
python scripts/view_evals.py --labels --top 5
```

**What it does:**
- Loads evaluation results from HF dataset (`YOUR_ORG/water-conflict-classifier-evals`)
- Displays metrics, configs, and comparisons in tabular format
- Analyzes hyperparameter impact on performance
- Shows per-label precision, recall, F1 for each experiment
- Helps identify optimal training configurations

**Requirements:**
- HF authentication: `hf auth login`
- Evals dataset populated via `train_on_hf.py` runs

---

### `demo_versioning.py`
Create sample experiments to demonstrate the versioning system.

**Usage:**
```bash
python scripts/demo_versioning.py
```

Creates demo data in `demo_experiment_history.jsonl` to explore the versioning features without running actual training.

---

### Building negatives

Negatives are news headlines that are not water conflict. They are built in five steps; each step caches its API responses in `../data/` so re-runs only fetch or generate what is missing.

| Step | Script | Output |
|------|--------|--------|
| 1. Real titles | `fetch_serpapi_negatives.py` | `real_headlines.csv` |
| 2. Synthetic titles | `generate_synthetic_negatives.py` | `synthetic_headlines.csv` |
| 3. Hard negatives | `generate_hard_negatives.py` | `hard_negatives.csv` |
| 4. LLM judge | `verify_negatives.py` | `*_verified.csv` |
| 5. Pool | `build_negatives.py` | `negatives_updated.csv`, `holdout_negatives.csv` |

The water conflict definition both LLM steps use lives in `negatives_common.py`.

**Requirements:** `SERPAPI_KEY` and `ANTHROPIC_API_KEY` in the project `.env` (gitignored).

### `fetch_serpapi_negatives.py`
Collect real news titles from Google News via [SerpApi](https://serpapi.com/google-news-api).

```bash
python fetch_serpapi_negatives.py --dry-run   # plan and search cost, no API calls
python fetch_serpapi_negatives.py             # run the plan
```

- Topic sections (World, Business, Technology, Entertainment, Sports, Science, Health) for the US, UK, India and Nigeria editions: ~200-300 titles per search, all from the past week
- Date-ranged keyword searches (`after:` / `before:`) spread across 2024-2026: ~100 titles per search
- One SerpApi search per entry (~51 in total); aborts if a run would leave fewer than 50 searches on the account
- Water-related titles are kept and marked; the judge decides which are safe

### `generate_synthetic_negatives.py`
Generate headlines with Claude Haiku 5.5.

```bash
uv run generate_synthetic_negatives.py --calls 140   # ~3,500 headlines
```

- Each call draws a category (conflict without water 40%, water without conflict 30%, general 30%), a subtopic, four countries and a headline style, plus 8 real titles as style examples
- Calls run in parallel (`--workers`, default 32); throughput is capped by the org's output-token rate limit

### `generate_hard_negatives.py`
Write 100 hand-written peaceful water headlines to `../data/hard_negatives.csv`.

### `verify_negatives.py`
LLM judge (Claude Haiku 5.5). Labels each headline `water_conflict`, `not_water_conflict` or `unsure`, and whether it is water-related. Only `not_water_conflict` headlines become negatives.

```bash
uv run verify_negatives.py ../data/real_headlines.csv ../data/synthetic_headlines.csv ../data/hard_negatives.csv
```

Any headline with both a water element and violence, protest or threat is never labelled `not_water_conflict`. On a sample of 50 positives, the judge passed none as safe.

### `build_negatives.py`
Build the training-ready pool.

```bash
uv run build_negatives.py
```

1. Keeps `not_water_conflict` headlines
2. Drops near-duplicates (bge-small embeddings, cosine ≥ 0.85; real titles win ties)
3. Sets aside headlines too close to any positive (`negatives_review.csv`)
4. Holds out 300 real + 300 synthetic negatives (`holdout_negatives.csv`), never used in training
5. Selects as many negatives as positives, 35% water-related, round-robin across k-means topic clusters

**Output:** `negatives_updated.csv` with `Headline`, `Basis`, `priority_sample` (True for water-related hard negatives) and `source` (`real` / `synthetic`).

---

### `upload_datasets.py`
Upload source datasets to Hugging Face Hub (raw data for reference).

**Usage:**
```bash
cd scripts
python upload_datasets.py
```

**What it does:**
- Loads `../data/positives.csv`
- Loads `../data/negatives_updated.csv` and `../data/holdout_negatives.csv` (from `build_negatives.py`)
- Creates/updates HF Hub dataset repository
- Uploads files to `YOUR_ORG/water-conflict-source-data` (source dataset - raw, unprocessed)
- Shows dataset composition (hard vs other negatives, real vs synthetic)

**Requirements:**
- HF authentication: `hf auth login`
- Config file: Copy `config.sample.py` to `config.py` and set `HF_ORGANIZATION`
- Negatives must be built first (see [Building negatives](#building-negatives))

**Note:** This uploads the RAW source data. Run `prepare_training_dataset.py` to create the training-ready dataset.

---

### `prepare_training_dataset.py`
Prepare and upload training-ready dataset to HuggingFace Hub.

**Usage:**
```bash
cd scripts
python prepare_training_dataset.py

# Custom configuration
python prepare_training_dataset.py --sample-size 1500 --test-size 0.2

# Use all data (no sampling)
python prepare_training_dataset.py --no-sampling

# Full control
python prepare_training_dataset.py \
  --sample-size 1200 \
  --test-size 0.15 \
  --random-state 42
```

**Arguments:**
- `--sample-size N`: Positive sample budget, split across labels (default: 1200)
- `--negative-ratio R`: Negatives added per sampled positive (default: 1.0)
- `--test-size F`: Test set proportion 0-1 (default: 0.15)
- `--random-state N`: Random seed for reproducibility (default: 42)
- `--no-sampling`: Use all data without sampling

**What it does:**
- Loads raw SOURCE data from HF Hub (`org/water-conflict-source-data`)
- Preprocesses: converts to multi-label format, balances hard negatives
- Splits into train/test sets (default: 85/15 split)
- Samples positives per label (~1,000 from a 1,200 budget), then adds an equal number of negatives (~2,000 examples)
- Copies `holdout_negatives.csv` from the source repo (used by `train_on_hf.py` to report false positive rates on real vs synthetic negatives)
- Uploads to `YOUR_ORG/water-conflict-training-data` as training-ready dataset

**Output:** Creates training-ready dataset with:
- `train.csv` - ready to train (no preprocessing needed)
- `test.csv` - ready to evaluate
- README with dataset details and configuration used

**Requirements:**
- HF authentication: `hf auth login`
- Source dataset uploaded (see `upload_datasets.py`)
- Config file with `HF_ORGANIZATION` set

**Why needed:** The training script just loads and trains - all preprocessing happens here, making training simpler and more reproducible.

---

### `train_on_hf.py`
Train the water conflict classifier on HuggingFace Jobs (managed GPUs).

**Usage:**
```bash
# From repo root
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=your-org \
  --namespace your-org \
  scripts/train_on_hf.py

# Optional: specify version explicitly
MODEL_VERSION=v2.0 hf jobs uv run ... scripts/train_on_hf.py
```

**What it does:**
- Loads training-ready data from HF Hub (`org/water-conflict-training-data`)
- Trains SetFit model on GPU infrastructure (data already preprocessed, balanced, split)
- Evaluates on held-out test set
- Pushes trained model to HF Hub with model card
- Auto-versions and logs experiment to `experiment_history.jsonl`
- Creates git tag on HF Hub for version retrieval
- Uploads evaluation results to HF evals dataset for comparison

**Requirements:**
- Package published to PyPI (already done: [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/))
- HF authentication token (set as `--secrets HF_TOKEN`)
- **Training dataset prepared** (see `prepare_training_dataset.py`)
- Config file with `HF_ORGANIZATION` set

**Configuration Options:**
- `--flavor a10g-large`: GPU type (recommend A10G for this task)
- `--timeout 2h`: Max runtime before auto-termination
- `--secrets HF_TOKEN`: Authentication token
- `--env HF_ORGANIZATION=yourorg`: Your HF org/username
- `--namespace yourorg`: Runs job under org account for billing
- `--env NUM_EPOCHS=2`: Training epochs (default: 1)
- `--env SAMPLING_STRATEGY=oversampling`: SetFit pair sampling (default: undersampling)
- `--env PUSH_TO_HUB=false`: Evaluate and log to `experiment_history.jsonl` only, for local sweeps

When the training dataset has `holdout_negatives.csv`, the run also reports the false positive rate on held-out real and synthetic negatives (model card and evals dataset).

**Expected Runtime:** ~2-5 minutes on A10G GPU

**Monitoring:**
```bash
# List jobs
hf jobs ps -a --namespace yourorg

# Stream logs
hf jobs logs <job_id> --namespace yourorg

# Cancel job
hf jobs cancel <job_id> --namespace yourorg
```

**Local Testing:**
```bash
# Test locally before submitting to HF Jobs
uv run scripts/train_on_hf.py
```

Note: Still requires dataset on HF Hub and proper authentication.

---

### `train_on_hf_optuna.py`
Train with Optuna hyperparameter search to find optimal settings automatically.

**Usage:**
```bash
# Default: 50 trials, search on 200 samples, train final on full data
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=your-org \
  --namespace your-org \
  scripts/train_on_hf_optuna.py

# Quick search (20 trials)
hf jobs uv run ... --env N_TRIALS=20 scripts/train_on_hf_optuna.py

# Search across multiple base models
hf jobs uv run ... --env SEARCH_MODELS=true --env N_TRIALS=100 scripts/train_on_hf_optuna.py
```

**Environment Variables:**
- `N_TRIALS`: Number of Optuna trials (default: 50)
- `SEARCH_SAMPLE_SIZE`: Samples for HP search (default: 200)
- `SEARCH_MODELS`: Search across base models (default: false)

**Strategy:**
1. Searches hyperparameters on small sample (~200) for fast trials
2. Trains final model on full dataset with best hyperparameters
3. Logs results + best params to HF evals dataset

**Hyperparameters searched:** learning rates, epochs, batch size, iterations, sampling strategy, LogisticRegression solver/max_iter.

---

### `distill_to_static.py`
Create a static (non-neural) version of your trained model for **50-500x faster inference** with no GPU.

**Usage:**
```bash
# Distill from HuggingFace Hub
uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier

# Custom output directory
uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --output ./my_static_model

# Adjust embedding dimensions (lower = faster, higher = more accurate)
uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --dims 384

# Run speed comparison test
uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --test
```

**What it does:**
- Loads your trained SetFit model from HF Hub
- Uses [model2vec](https://github.com/MinishLab/model2vec) to distill embeddings to static vectors
- Converts neural network embeddings → simple word vector lookups (no matrix multiplication)
- Keeps your trained classification heads intact
- Saves complete model to disk for deployment
- Shows before/after speed comparison

**How it works:**
```python
from model2vec import distill

# Distill embedding layer to static vectors
static_embeddings = distill(
    model_name=your_setfit_model.model_body.model_name_or_path,
    pca_dims=256  # Reduce dimensionality via PCA
)

# Keep trained classification heads
heads = {label: model.model_head.estimators_[i] 
         for i, label in enumerate(model.labels)}

# Inference: embeddings → predictions
embeddings = static_embeddings.encode(texts)
predictions = {label: head.predict(embeddings) for label, head in heads.items()}
```

**Output structure:**
```
static_model/
├── embeddings/           # Static word vectors (model2vec format)
│   ├── model.safetensors
│   └── config.json
├── heads.pkl             # Your trained classification heads
└── config.json           # Model metadata
```

**Benefits:**
- **50-500x faster** than original SetFit model
- **No GPU required** - runs on CPU efficiently
- **Smaller model size** - just vocabulary + vectors
- **Same accuracy** - classification heads unchanged
- **Zero dependencies** on PyTorch/TensorFlow at inference

**Trade-offs:**
- Slight accuracy drop (~1-3%) from PCA compression
- Static embeddings can't adapt to context (but SetFit doesn't use context much anyway)
- Best for high-throughput scenarios (batch processing, real-time APIs)

**When to use:**
- Production APIs with high request volume
- Batch processing millions of headlines
- Edge deployment (embedded systems, mobile)
- Cost optimization (cheaper CPU vs GPU instances)

**Requirements:**
- Trained SetFit model on HF Hub (see `train_on_hf.py`)
- `pip install model2vec setfit sentence-transformers`

**Expected distillation time:** ~30-60 seconds

---

### `inference_static.py`
Quick inference using the distilled static model.

**Usage:**
```bash
# Single prediction
uv run scripts/inference_static.py "Military group attack workers at dam"

# Multiple predictions
uv run scripts/inference_static.py \
  "Military group attacks workers at the Kajaki Dam" \
  "New water treatment plant opens in California"

# Specify model path
uv run scripts/inference_static.py "Text" --model ./my_static_model
```

**Example output:**
```
Text: Military group attack workers at the Kajaki Dam
Labels: Trigger, Casualty, Weapon

Text: New water treatment plant opens in California
Labels: None (not a water conflict)
```

**What it does:**
- Loads static model (embeddings + classification heads)
- Runs fast inference on provided texts
- Shows detected conflict labels

**Requirements:**
- Static model created via `distill_to_static.py`
- `pip install model2vec`

---

## Typical Workflow

Choose your path based on what you want to do:

### For Cloud Training (HF Jobs) - RECOMMENDED

```bash
# 1-2. Build negatives (if needed) - see "Building negatives"
cd scripts
python fetch_serpapi_negatives.py
uv run generate_synthetic_negatives.py --calls 140
python generate_hard_negatives.py
uv run verify_negatives.py ../data/real_headlines.csv ../data/synthetic_headlines.csv ../data/hard_negatives.csv
uv run build_negatives.py

# 3. Configure your HF organization
cd ..
cp config.sample.py config.py
# Edit config.py and set HF_ORGANIZATION

# 4. Upload SOURCE data to HF Hub (raw data for reference)
cd scripts
python upload_datasets.py
# Uploads positives, negatives_updated.csv (as negatives.csv) and holdout_negatives.csv

# 5. Prepare TRAINING dataset (preprocessed, balanced, split, sampled)
python prepare_training_dataset.py
# Samples ~1,000 positives (balanced per label) plus an equal number of negatives

# 6. Run training on HF Jobs
cd ..
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=yourorg \
  --namespace yourorg \
  scripts/train_on_hf.py
```

**Key Point:** Steps 4-5 create two datasets:
- **SOURCE** (`water-conflict-source-data`): Raw positives + negatives for reference
- **TRAINING** (`water-conflict-training-data`): Preprocessed, ready-to-train data

The training script (step 6) simply loads the TRAINING dataset and trains - no preprocessing needed.

### For Local Training

```bash
# 1. Build negative examples (see "Building negatives")
cd scripts
uv run build_negatives.py

# 2. Train locally
cd ../classifier
python train_setfit_headline_classifier.py
```

### For Static Model Distillation and Usage

After training, create a fast static version:

```bash
# 1. Distill trained model to static embeddings
uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --test

# 2. Test the static model
uv run scripts/inference_static.py "Military group attack workers at the Kajaki Dam in Afghanistan"

# Or use in Python code:
python -c "
from model2vec import StaticModel
import pickle

# Load
embeddings = StaticModel.from_pretrained('./static_model/embeddings')
with open('./static_model/heads.pkl', 'rb') as f:
    heads = pickle.load(f)

# Predict
texts = ['Military group attack workers at the Kajaki Dam in Afghanistan']
emb = embeddings.encode(texts)
predictions = {label: head.predict(emb) for label, head in heads.items()}
print(predictions)
"

# 3. Deploy static model to production
# - Copy static_model/ to your server
# - No GPU needed, runs on CPU
# - 50-500x faster than original model
```

## Note

These scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package. For package source code and local training, see `../classifier/`
