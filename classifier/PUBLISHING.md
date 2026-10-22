# Publishing the Package

## Project Structure

This is a **mono repo** where `/classifier` is the publishable package:

```
waterconflict/                    # Mono repo root
├── data/                         # Training data
├── config.py                     # Project config
├── classifier/                   # 📦 PACKAGE ROOT (published to PyPI)
│   ├── pyproject.toml           # Package definition
│   ├── requirements.txt         # Dependencies
│   ├── setup.py                 # Build config
│   ├── __init__.py              # Package marker
│   ├── data_prep.py            # 🔧 Module
│   ├── training_logic.py       # 🔧 Module
│   ├── evaluation.py           # 🔧 Module
│   ├── model_card.py           # 🔧 Module
│   ├── train_setfit_headline_classifier.py  # Local training script
│   └── README.md               # Package docs
└── scripts/                      # Utility scripts (uses published package)
    ├── upload_datasets.py       # Upload training data to HF Hub
    ├── build_negatives.py       # Build the negatives pool (dedupe, diversify, hold out)
    └── train_on_hf.py          # HF Jobs training (uses PyPI package)
```

**Important:** `/scripts/` folder is separate! It contains utility scripts that **use** the published package:
- `upload_datasets.py` - Upload training data to HF Hub
- `build_negatives.py` - Build the negatives pool from verified real and synthetic headlines
- `train_on_hf.py` - Cloud training script (uses `water-conflict-classifier` from PyPI)
- These are NOT part of the package (and shouldn't be)

---

## Building with UV

```bash
cd classifier

# Build the package (creates dist/ folder)
uv build

# This creates:
# - dist/water-conflict-classifier-0.1.0.tar.gz
# - dist/water-conflict-classifier-0.1.0-py3-none-any.whl
```

## Publishing with Twine

### First Time Setup

1. Install twine:
   ```bash
   pip install twine
   ```

2. Create accounts and tokens:
   - **TestPyPI**: https://test.pypi.org/account/register/
   - **Production PyPI**: https://pypi.org/account/register/
   - Generate API tokens from: https://test.pypi.org/manage/account/token/ and https://pypi.org/manage/account/token/

3. Configure `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-production-token-here

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-test-token-here
   ```

### Publish to Test PyPI (Recommended First)

```bash
cd classifier

# Build first
uv build

# Publish to test PyPI (using twine)
twine upload --repository testpypi dist/*

# Test installation from test PyPI (with fallback to PyPI for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ water-conflict-classifier
```

### Publish to Production PyPI

```bash
cd classifier

# Make sure you've tested on test PyPI first!
uv build

# Publish to real PyPI (using twine)
twine upload dist/*

# Now anyone can install:
# pip install water-conflict-classifier
```

---

## Complete Workflow

```bash
# 1. Navigate to package
cd classifier

# 2. Install locally for development
uv pip install -e .

# 3. Test everything works
python train_setfit_headline_classifier.py  # Test local training

# 4. Update version in pyproject.toml if needed
# version = "0.1.1"  # Bump version

# 5. Build
uv build

# 6. Publish to Test PyPI
twine upload --repository testpypi dist/*

# 7. Test from test PyPI (with fallback to PyPI for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ water-conflict-classifier==0.1.0

# 8. If all good, publish to real PyPI
twine upload dist/*

# 9. Now HF Jobs can use it!
cd ..  # Back to repo root
hf jobs uv run \
  --flavor a10g-large \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=yourorg \
  --namespace yourorg \
  scripts/train_on_hf.py
```

---

## Alternative: Use Git URL (No Publishing Needed)

If you don't want to publish to PyPI, you can use a Git URL in your UV dependencies:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add classifier package"
   git push
   ```

2. **Update scripts/train_on_hf.py dependencies:**
   ```python
   # /// script
   # dependencies = [
   #     "water-conflict-classifier @ git+https://github.com/yourusername/waterconflict.git#subdirectory=classifier",
   # ]
   # ///
   ```

3. **Works immediately with HF Jobs:**
   ```bash
   hf jobs uv run \
     --flavor a10g-large \
     --secrets HF_TOKEN \
     scripts/train_on_hf.py
   ```

---

## Mono Repo Organization

**Keep separate:**
- `/classifier/` - The package (training code) - **PUBLISH THIS**
- `/scripts/` - Utility scripts (data prep, upload) - **DON'T PUBLISH**
- `/data/` - Training data

**Optional cleanup:**
If you have an old `/water_conflict_classifier/` folder (duplicate), you can delete it.

---

## Troubleshooting

### "Module not found" errors

Make sure you're running from the right directory:
```bash
cd classifier  # Must be in package root
uv pip install -e .
python train_setfit_headline_classifier.py
```

### Twine authentication issues

If you get 403 errors, make sure your `~/.pypirc` is configured correctly with the API tokens.

You can also pass credentials directly:
```bash
twine upload --repository testpypi dist/* --username __token__ --password pypi-your-token-here
```

### Version conflicts

Always bump version in `pyproject.toml` before publishing:
```toml
version = "0.1.1"  # Increment this
```

PyPI doesn't allow re-uploading the same version.
