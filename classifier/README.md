# Water Conflict Classifier

SetFit-based multi-label text classifier for identifying water-related conflict events in news headlines.

**This folder contains the package source code.** For usage instructions with the published package, see the [PyPI page](https://pypi.org/project/water-conflict-classifier/).

**Project:** Experimental research supporting the [Pacific Institute's Water Conflict Chronology](https://www.worldwater.org/water-conflict/)  
**Developer:** Baobab Tech  
**License:** [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial)  
**PyPI Package:** [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/)

---

## Package Installation & Usage

Install from PyPI:

```bash
pip install water-conflict-classifier
```

Use the trained model:

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("baobabtech/water-conflict-classifier")
predictions = model.predict(["Taliban attack workers at dam"])
# Returns: [[1, 1, 1]]  # [Trigger, Casualty, Weapon]
```

**The rest of this README is for developers who want to train their own model or modify the package.**

## Frugal AI: Training with Limited Data

This classifier demonstrates an intentional approach to building AI systems with **limited data** using [SetFit](https://huggingface.co/docs/setfit/en/index) - a framework for few-shot learning with sentence transformers. Rather than defaulting to massive language models (GPT, Claude, or 100B+ parameter models) for simple classification tasks, we fine-tune small, efficient models (e.g., BAAI/bge-small-en-v1.5 with ~33M parameters) on a focused dataset.

**Why this matters:** The industry has normalized using trillion-parameter models to classify headlines, answer simple questions, or categorize text - tasks that don't require world knowledge, reasoning, or generative capabilities. This is computationally wasteful and environmentally costly. A properly fine-tuned small model can achieve comparable or better accuracy while using a fraction of the compute resources.

**Our approach:**
- Train on ~600 examples (few-shot learning with SetFit)
- Deploy small parameter models (e.g., ~33M params) vs. 100B-1T parameter alternatives
- Achieve specialized task performance without the overhead of general-purpose LLMs
- Reduce inference costs and latency by orders of magnitude

This is not about avoiding large models altogether - they're invaluable for complex reasoning tasks. But for targeted classification problems with labeled data, fine-tuning remains the professional, responsible choice.

## Package Structure

This is the **source code** for the `water-conflict-classifier` Python package, published to PyPI.

```
classifier/
├── __init__.py                         # Package marker
├── data_prep.py                        # Data loading & preprocessing
├── training_logic.py                   # Core training logic
├── evaluation.py                       # Model evaluation & metrics
├── model_card.py                       # Model card generation
├── train_setfit_headline_classifier.py # Local training script
├── pyproject.toml                      # Package configuration
├── setup.py                            # Build configuration
└── README.md                           # This file
```

**Note:** Scripts that use this package (like cloud training with HF Jobs) are in the `../scripts/` folder.

---

## Local Training

Train on your own hardware with local data files.

### Setup

1. **Install the package in development mode:**

```bash
uv pip install -e .
# or with regular pip:
pip install -e .
```

This installs the modules (`data_prep`, `training_logic`, etc.) so they can be imported.

2. **Prepare training data:**

Training data should be in `../data/`:
- `../data/positives.csv` - Water conflict headlines with labels
- `../data/negatives.csv` - Non-water conflict headlines

Generate negatives from ACLED (if needed):
```bash
cd ../scripts
python transform_prep_negatives.py
```

3. **Train:**

```bash
python train_setfit_headline_classifier.py
```

Model saved to `./water-conflict-classifier/`

---

## Cloud Training

For training on HuggingFace Jobs (managed GPUs), see `../scripts/train_on_hf.py` and the [scripts README](../scripts/README.md).

---

## Publishing to PyPI

See [PUBLISHING.md](PUBLISHING.md) for complete instructions on building and publishing the package.

---

## Data Sources

The training data combines:

- **Positive Examples**: Water conflict headlines from [Pacific Institute Water Conflict Chronology](https://www.worldwater.org/water-conflict/)
- **Negative Examples**: Two types for balanced training:
  1. **Hard Negatives (~120)**: Water-related peaceful news (infrastructure, research, conservation) to prevent false positives
  2. **ACLED Negatives (~600)**: Non-water conflict events from [ACLED](https://acleddata.com/)

### Hard Negatives Strategy

Without hard negatives, the model learns "water mentioned → conflict" instead of "water + violence → conflict". Hard negatives are water-related headlines that lack violence or conflict:

- Water infrastructure projects (dams, treatment plants)
- Scientific water research and technology
- Water conservation initiatives and conferences  
- Environmental water management

These are tagged with `priority_sample=True` in the dataset and are ALWAYS included in training (never diluted by sampling). This ensures the model correctly distinguishes peaceful water news from actual water conflicts.

## Resources

- [HF Jobs Guide](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [UV Script Format](https://docs.astral.sh/uv/guides/scripts/) (used in `train_on_hf.py`)
- [SetFit Documentation](https://huggingface.co/docs/setfit)
- [Pacific Institute Water Conflict Chronology](https://www.worldwater.org/water-conflict/)
- [ACLED Data](https://acleddata.com/)

---

## License

Copyright © 2025 Baobab Tech

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

You are free to use, share, and adapt this work for non-commercial purposes with appropriate attribution to Baobab Tech. For commercial licensing inquiries, please contact Baobab Tech.
