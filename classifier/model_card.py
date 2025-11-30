"""
Model card generation for water conflict classifier.

Creates comprehensive model cards with training details and evaluation metrics.
"""

import requests
from typing import Optional


def get_base_model_info(model_id: str) -> dict:
    """
    Fetch base model information from HuggingFace API.
    
    Args:
        model_id: HuggingFace model ID (e.g. 'sentence-transformers/all-mpnet-base-v2')
        
    Returns:
        Dictionary with 'params' (int) and 'size_mb' (int) keys
    """
    try:
        url = f"https://huggingface.co/api/models/{model_id}/revision/main"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract parameter count from safetensors info
        total_params = 0
        if 'safetensors' in data and 'parameters' in data['safetensors']:
            total_params = data['safetensors'].get('total', 0)
        
        # Format parameter count (e.g., 109486978 -> "109M")
        params_formatted = f"{total_params / 1_000_000:.0f}M" if total_params > 0 else "33M"
        
        # Estimate size in MB (rough estimate: 4 bytes per param for float32)
        size_mb = int(total_params * 4 / 1_000_000) if total_params > 0 else 130
        
        return {
            'params': params_formatted,
            'params_raw': total_params,
            'size_mb': size_mb
        }
    except Exception as e:
        print(f"Warning: Could not fetch model info from HF API: {e}")
        # Fallback to defaults
        return {
            'params': '33M',
            'params_raw': 33_000_000,
            'size_mb': 130
        }


def generate_model_card(model_repo: str,
                       base_model: str,
                       label_names: list[str],
                       eval_results: dict,
                       train_size: int,
                       test_size: int,
                       batch_size: int,
                       num_epochs: int,
                       num_iterations: int = 20,
                       sampling_strategy: str = "undersampling",
                       evals_repo: Optional[str] = None,
                       training_dataset_repo: Optional[str] = None,
                       dataset_version: Optional[str] = None) -> str:
    """
    Generate comprehensive model card with evaluation results.
    
    Args:
        model_repo: HF model repository ID
        base_model: Base model used for training
        label_names: List of label names
        eval_results: Dictionary from evaluate_model()
        train_size: Number of training examples
        test_size: Number of test examples
        batch_size: Training batch size
        num_epochs: Number of training epochs
        num_iterations: Number of contrastive pair iterations
        sampling_strategy: Training sampling strategy
        evals_repo: Optional HF evals dataset repo ID for experiment tracking
        training_dataset_repo: Optional HF dataset repo ID for versioned training data
        dataset_version: Optional dataset version (e.g., "d1.0")
        
    Returns:
        Model card markdown string
    """
    overall = eval_results['overall']
    per_label = eval_results['per_label']
    
    # Fetch base model info from HF API
    model_info = get_base_model_info(base_model)
    
    # Calculate test split percentage
    test_split = test_size / (train_size + test_size)
    
    # Training info
    training_info = f"{train_size} examples"
    
    model_card = f"""---
license: cc-by-nc-4.0
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- multi-label
- water-conflict
metrics:
- f1
- accuracy
language:
- en
widget:
- text: "Military attack workers at the Kajaki Dam in Afghanistan"
- text: "Violent protests erupt over dam construction in Sudan"
- text: "New water treatment plant opens in California"
- text: "Armed groups cut off water supply to villages in Syria"
- text: "Government announces new irrigation subsidies"
---

# Water Conflict Multi-Label Classifier

## 🔬 Experimental Research

> This experimental research draws on Pacific Institute's [Water Conflict Chronology](https://www.worldwater.org/water-conflict/), which tracks water-related conflicts spanning over 4,500 years of human history. The work is conducted independently and is not affiliated with Pacific Institute.

This model is designed to assist researchers in classifying water-related conflict events at scale using tiny/small models that can classify 100s of headlines per second.

The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history. This is not a commercial product and is not intended for commercial use.

## 📋 Model Description

This SetFit-based model classifies news headlines about water-related conflicts into three categories:

- **Trigger**: Water resource as a conflict trigger
- **Casualty**: Water infrastructure as a casualty/target  
- **Weapon**: Water used as a weapon/tool

These categories align with the Pacific Institute's Water Conflict Chronology framework for understanding how water intersects with security and conflict.

## 🏗️ Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Architecture**: SetFit with One-vs-Rest multi-label strategy
- **Training Approach**: Few-shot learning optimized (SetFit reaches peak performance with small samples)
- **Training samples**: {training_info}
- **Test samples**: {test_size} (held-out, never seen during training)
- **Training time**: ~2-5 minutes on A10G GPU
- **Model size**: {model_info['params']} Parameters, ~{model_info['size_mb']}MB
- **Inference speed**: ~5-10ms per headline on CPU

## 💻 Usage

### Quick Start

```python
from setfit import SetFitModel

# Load the trained model from HF Hub
model = SetFitModel.from_pretrained("{model_repo}")

# Predict on headlines
headlines = [
    "Military attack workers at the Kajaki Dam in Afghanistan",
    "New water treatment plant opens in California"
]

predictions = model.predict(headlines)
print(predictions)
# Output: [[1, 1, 0], [0, 0, 0]]
# Format: [Trigger, Casualty, Weapon]
```

### Interpreting Results

The model returns a list of binary predictions for each label:

```python
label_names = ['Trigger', 'Casualty', 'Weapon']

for headline, pred in zip(headlines, predictions):
    labels = [label_names[i] for i, val in enumerate(pred) if val == 1]
    print(f"Headline: {{headline}}")
    print(f"Labels: {{', '.join(labels) if labels else 'None'}}")
    print()
```

### Batch Processing

```python
import pandas as pd

# Load your data
df = pd.read_csv("your_headlines.csv")

# Predict in batches
predictions = model.predict(df['headline'].tolist())

# Add predictions to dataframe
df['trigger'] = [p[0] for p in predictions]
df['casualty'] = [p[1] for p in predictions]
df['weapon'] = [p[2] for p in predictions]
```

### Example Outputs

| Headline | Trigger | Casualty | Weapon |
|----------|---------|----------|--------|
| "Armed groups blow up water pipeline in Iraq" | ✓ | ✓ | ✓ |
| "New water treatment plant opens in California" | ✗ | ✗ | ✗ |
| "Protests erupt over dam construction in Ethiopia" | ✓ | ✗ | ✗ |

## 📈 Evaluation Results

Evaluated on a held-out test set of {test_size} samples ({test_split*100:.0f}% of total data, stratified by label combinations).

### Overall Performance

| Metric | Score |
|--------|-------|
| Exact Match Accuracy | {overall['accuracy']:.4f} |
| Hamming Loss | {overall['hamming_loss']:.4f} |
| F1 (micro) | {overall['f1_micro']:.4f} |
| F1 (macro) | {overall['f1_macro']:.4f} |
| F1 (samples) | {overall['f1_samples']:.4f} |

### Per-Label Performance

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
"""
    
    for label_name in label_names:
        metrics = per_label[label_name]
        model_card += f"| {label_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |\n"
    
    model_card += f"""
### Training Details

- **Training samples**: {train_size} examples
- **Test samples**: {test_size} examples (held-out before sampling)
- **Base model**: {base_model} ({model_info['params']} params)
- **Batch size**: {batch_size}
- **Epochs**: {num_epochs}
- **Iterations**: {num_iterations} (contrastive pair generation)
- **Sampling strategy**: {sampling_strategy} (balances positive/negative pairs)
"""
    
    # Add training dataset link if provided
    if training_dataset_repo:
        dataset_version_str = f" (version: {dataset_version})" if dataset_version else ""
        model_card += f"""- **Training Dataset**: [{training_dataset_repo}](https://huggingface.co/datasets/{training_dataset_repo}){dataset_version_str}

"""
    
    # Add experiment tracking section if evals repo is provided
    if evals_repo:
        model_card += f"""
### 📈 Experiment Tracking

All training runs are automatically tracked in a public dataset for experiment comparison:

- **Evals Dataset**: [{evals_repo}](https://huggingface.co/datasets/{evals_repo})
- **Tracked Metrics**: F1 scores, accuracy, per-label performance, and all hyperparameters
- **Compare Experiments**: View how different configurations (sample size, epochs, batch size) affect performance
- **Reproducibility**: Full training configs logged for each version

You can explore past experiments and compare model performance across versions using the evals dataset.

"""
    
    model_card += """
## 📊 Data Sources

### Positive Examples (Water Conflict Headlines)
Pacific Institute (2025). *Water Conflict Chronology*. Pacific Institute, Oakland, CA.  
https://www.worldwater.org/water-conflict/

### Negative Examples (Non-Water Conflict Headlines)
Armed Conflict Location & Event Data Project (ACLED).  
https://acleddata.com/

**Note:** Training negatives include synthetic "hard negatives" - peaceful water-related news (e.g., "New desalination plant opens", "Water conservation conference") to prevent false positives on non-conflict water topics.

## 🌍 About This Project

This model is part of independent experimental research drawing on the Pacific Institute's Water Conflict Chronology. The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history.

**Project Links:**
- Pacific Institute Water Conflict Chronology: https://www.worldwater.org/water-conflict/
- Python Package (PyPI): https://pypi.org/project/water-conflict-classifier/
- Source Code: https://github.com/baobabtech/waterconflict
- Model Hub: https://huggingface.co/{model_repo}


## 🌱 Frugal AI: Training with Limited Data

This classifier demonstrates an intentional approach to building AI systems with **limited data** using [SetFit](https://huggingface.co/docs/setfit/en/index) - a framework for few-shot learning with sentence transformers. Rather than defaulting to massive language models (GPT, Claude, or 100B+ parameter models) for simple classification tasks, we fine-tune small, efficient models (e.g., BAAI/bge-small-en-v1.5 with ~33M parameters) on a focused dataset.

**Why this matters:** The industry has normalized using trillion-parameter models to classify headlines, answer simple questions, or categorize text - tasks that don't require world knowledge, reasoning, or generative capabilities. This is computationally wasteful and environmentally costly. A properly fine-tuned small model can achieve comparable or better accuracy while using a fraction of the compute resources.

**Our approach:**
- Train on ~600 examples (few-shot learning with SetFit)
- Deploy small parameter models (e.g., ~33M params) vs. 100B-1T parameter alternatives
- Achieve specialized task performance without the overhead of general-purpose LLMs
- Reduce inference costs and latency by orders of magnitude

This is not about avoiding large models altogether - they're invaluable for complex reasoning tasks. But for targeted classification problems with labeled data, fine-tuning remains the professional, responsible choice.


### 🏋🏽‍♀️ Training Your Own Model

You can train your own version using the [published package](https://pypi.org/project/water-conflict-classifier/).

**Package includes:**
- Data preprocessing utilities
- Training logic (SetFit multi-label)
- Evaluation metrics
- Model card generation

**Source code:** https://github.com/baobabtech/waterconflict/tree/main/classifier  
**PyPI:** https://pypi.org/project/water-conflict-classifier/

```bash
# Install package
pip install water-conflict-classifier

# Or install from source for development
git clone https://github.com/baobabtech/waterconflict.git
cd waterconflict/classifier
pip install -e .

# Train locally
python train_setfit_headline_classifier.py
```

For cloud training on HuggingFace Jobs infrastructure, see the scripts folder in the repository.

## 📜 License

Copyright © 2025 Baobab Tech

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit to Baobab Tech, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes


## 📝 Citation

If you use this model in your work, please cite:

```bibtex
@misc{{waterconflict2025,
  title={{Water Conflict Multi-Label Classifier}},
  author={{Independent Experimental Research Drawing on Pacific Institute Water Conflict Chronology}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{model_repo}}}}},
  note={{Training data from Pacific Institute Water Conflict Chronology and ACLED}}
}}
```

Please also cite the Pacific Institute's Water Conflict Chronology:

```bibtex
@misc{{pacificinstitute2025,
  title={{Water Conflict Chronology}},
  author={{Pacific Institute}},
  year={{2025}},
  address={{Oakland, CA}},
  url={{https://www.worldwater.org/water-conflict/}},
  note={{Accessed: [access date]}}
}}
```

"""
    
    return model_card

