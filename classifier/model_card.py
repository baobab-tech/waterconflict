"""
Model card generation for water conflict classifier.

Creates comprehensive model cards with training details and evaluation metrics.
"""


def generate_model_card(model_repo: str,
                       base_model: str,
                       label_names: list[str],
                       eval_results: dict,
                       train_size: int,
                       test_size: int,
                       batch_size: int,
                       num_epochs: int,
                       test_split: float,
                       full_train_size: int = None,
                       sampling_strategy: str = "oversampling") -> str:
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
        test_split: Test set proportion (e.g. 0.15)
        full_train_size: Original training pool size (before sampling)
        sampling_strategy: Training sampling strategy
        
    Returns:
        Model card markdown string
    """
    overall = eval_results['overall']
    per_label = eval_results['per_label']
    
    # Handle sampling info
    training_info = f"{train_size} examples"
    if full_train_size and full_train_size > train_size:
        training_info = f"{train_size} (sampled from {full_train_size} total training pool)"
    
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
- text: "Taliban attack workers at the Kajaki Dam in Afghanistan"
- text: "Violent protests erupt over dam construction in Sudan"
- text: "New water treatment plant opens in California"
- text: "ISIS cuts off water supply to villages in Syria"
- text: "Government announces new irrigation subsidies"
---

# Water Conflict Multi-Label Classifier

## 🔬 Experimental Research

> **Note:** This is experimental research to potentially support the Pacific Institute's [Water Conflict Chronology](https://www.worldwater.org/water-conflict/) project, which tracks water-related conflicts spanning over 4,500 years of human history.

This model is designed to assist researchers in classifying water-related conflict events. 

The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history. This is not a commercial product and is not intended for commercial use.

## 🌱 Frugal AI: Training with Limited Data

This classifier demonstrates an intentional approach to building AI systems with **limited data** using [SetFit](https://huggingface.co/docs/setfit/en/index) - a framework for few-shot learning with sentence transformers. Rather than defaulting to massive language models (GPT, Claude, or 100B+ parameter models) for simple classification tasks, we fine-tune a small, efficient model (~33M parameters) on a focused dataset.

**Why this matters:** The industry has normalized using trillion-parameter models to classify headlines, answer simple questions, or categorize text - tasks that don't require world knowledge, reasoning, or generative capabilities. This is computationally wasteful and environmentally costly. A properly fine-tuned small model can achieve comparable or better accuracy while using a fraction of the compute resources.

**Our approach:**
- Train on ~600 examples (few-shot learning with SetFit)
- Deploy a 33M parameter model vs. 100B-1T parameter alternatives
- Achieve specialized task performance without the overhead of general-purpose LLMs
- Reduce inference costs and latency by orders of magnitude

This is not about avoiding large models altogether - they're invaluable for complex reasoning tasks. But for targeted classification problems with labeled data, fine-tuning remains the professional, responsible choice.

## 📋 Model Description

This SetFit-based model classifies news headlines about water-related conflicts into three categories:

- **Trigger**: Water resource as a conflict trigger
- **Casualty**: Water infrastructure as a casualty/target  
- **Weapon**: Water used as a weapon/tool

These categories align with the Pacific Institute's Water Conflict Chronology framework for understanding how water intersects with security and conflict.

## 🏗️ Model Details

- **Base Model**: {base_model} (33.4M parameters)
- **Architecture**: SetFit with One-vs-Rest multi-label strategy
- **Training Approach**: Few-shot learning optimized (SetFit reaches peak performance with small samples)
- **Training samples**: {training_info}
- **Test samples**: {test_size} (held-out, never seen during training)
- **Training time**: ~2-5 minutes on A10G GPU
- **Model size**: ~130MB
- **Inference speed**: ~5-10ms per headline on CPU

## 💻 Usage

### Installation

The training code is published as a Python package on PyPI:

```bash
pip install water-conflict-classifier
```

**Package includes:**
- Data preprocessing utilities
- Training logic (SetFit multi-label)
- Evaluation metrics
- Model card generation

**Source code:** https://github.com/baobabtech/waterconflict/tree/main/classifier  
**PyPI:** https://pypi.org/project/water-conflict-classifier/

### Quick Start

```python
from setfit import SetFitModel

# Load the trained model from HF Hub
model = SetFitModel.from_pretrained("{model_repo}")

# Predict on headlines
headlines = [
    "Taliban attack workers at the Kajaki Dam in Afghanistan",
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
| "ISIS militants blow up water pipeline in Iraq" | ✓ | ✓ | ✓ |
| "New water treatment plant opens in California" | ✗ | ✗ | ✗ |
| "Protests erupt over dam construction in Ethiopia" | ✓ | ✗ | ✗ |

## Evaluation Results

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
- **Base model**: {base_model} (33.4M params)
- **Batch size**: {batch_size}
- **Epochs**: {num_epochs}
- **Sampling strategy**: {sampling_strategy} (balances positive/negative pairs)

## 📊 Data Sources

### Positive Examples (Water Conflict Headlines)
Pacific Institute (2025). *Water Conflict Chronology*. Pacific Institute, Oakland, CA.  
https://www.worldwater.org/water-conflict/

### Negative Examples (Non-Water Conflict Headlines)
Armed Conflict Location & Event Data Project (ACLED).  
https://acleddata.com/

## 🌍 About This Project

This model is part of experimental research supporting the Pacific Institute's Water Conflict Chronology project. The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history.

**Project Links:**
- Pacific Institute Water Conflict Chronology: https://www.worldwater.org/water-conflict/
- Python Package (PyPI): https://pypi.org/project/water-conflict-classifier/
- Source Code: https://github.com/baobabtech/waterconflict
- Model Hub: https://huggingface.co/{model_repo}

### Training Your Own Model

You can train your own version using the published package:

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

For commercial licensing inquiries, please contact Baobab Tech.

## 📝 Citation

If you use this model in your work, please cite:

```bibtex
@misc{{waterconflict2025,
  title={{Water Conflict Multi-Label Classifier}},
  author={{Experimental Research Supporting Pacific Institute Water Conflict Chronology}},
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

**Recommended citation format:**  
Pacific Institute (2025) Water Conflict Chronology. Pacific Institute, Oakland, CA. https://www.worldwater.org/water-conflict/. Accessed: (access date).
"""
    
    return model_card

