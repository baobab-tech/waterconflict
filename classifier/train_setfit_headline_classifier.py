"""
SetFit Multi-Label Water Conflict Classifier Training Script (Local)

Inputs:
- ../data/negatives.csv: CSV with 'Headline' column (non-conflict news)
- ../data/positives.csv: CSV with 'Headline' and 'Basis' columns

Output:
- Trained SetFit model saved to ./water-conflict-classifier/
"""

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import shared modules
from data_prep import load_local_data, preprocess_data, LABEL_NAMES
from training_logic import train_model
from evaluation import evaluate_model, print_evaluation_results

print("=" * 80)
print("SetFit Multi-Label Water Conflict Classifier Training")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

positives, negatives = load_local_data(
    positives_path="../data/positives.csv",
    negatives_path="../data/negatives.csv"
)

# ============================================================================
# 2. PREPROCESS DATA
# ============================================================================
print("\n[2/6] Preprocessing data...")

data = preprocess_data(positives, negatives, balance_negatives=False)

# Label distribution
label_counts = np.array(data['labels'].tolist()).sum(axis=0)
print("\n  Label distribution in positives:")
for name, count in zip(LABEL_NAMES, label_counts):
    print(f"    - {name}: {int(count)} ({count/len(positives)*100:.1f}%)")

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================
print("\n[3/6] Splitting dataset...")

train_data_raw, test_data_raw = train_test_split(
    data, 
    test_size=0.15,  # 15% for testing
    random_state=42,
    stratify=data['labels'].apply(lambda x: tuple(x))  # Stratify by label combination
)

# Ensure DataFrames are properly typed
import pandas as pd
train_data: pd.DataFrame = pd.DataFrame(train_data_raw).reset_index(drop=True)
test_data: pd.DataFrame = pd.DataFrame(test_data_raw).reset_index(drop=True)

print(f"  ✓ Training set: {len(train_data)} examples")
print(f"  ✓ Test set: {len(test_data)} examples")

# Convert to HuggingFace Dataset format
train_df: pd.DataFrame = train_data[['text', 'labels']]  # type: ignore
test_df: pd.DataFrame = test_data[['text', 'labels']]  # type: ignore
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================
print("\n[4/6] Training model...")
print("  Base model: BAAI/bge-small-en-v1.5 (33.4M params)")
print("  This will take approximately 5-15 minutes depending on your hardware.\n")

model = train_model(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    base_model="BAAI/bge-small-en-v1.5",
    label_names=LABEL_NAMES,
    batch_size=16,
    num_epochs=3,
    sampling_strategy="oversampling"
)

# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================
print("\n[5/6] Evaluating model...")

eval_results = evaluate_model(
    model=model,
    test_texts=test_data['text'].tolist(),
    test_labels=test_data['labels'].tolist(),
    label_names=LABEL_NAMES
)

print_evaluation_results(eval_results, LABEL_NAMES)

# ============================================================================
# 6. SAVE MODEL
# ============================================================================
print("\n[6/6] Saving model...")
print("=" * 80)

model_path = "./water-conflict-classifier"
model.save_pretrained(model_path)

print(f"\n  ✓ Model saved to: {model_path}")
print("\n  To load the model later:")
print(f"    from setfit import SetFitModel")
print(f"    model = SetFitModel.from_pretrained('{model_path}')")

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)

test_examples = [
    "Taliban attack workers at the Kajaki Dam in Afghanistan",
    "New water treatment plant opens in California",
    "Violent protests erupt over dam construction in Sudan",
    "Scientists discover new water filtration method"
]

predictions = model.predict(test_examples)

print("\n")
for text, pred in zip(test_examples, predictions):  # type: ignore
    labels_detected = [LABEL_NAMES[i] for i, val in enumerate(pred) if val == 1]
    label_str = ", ".join(labels_detected) if labels_detected else "None"
    print(f"  Text: {text[:70]}...")
    print(f"  Labels: {label_str}")
    print()

print("=" * 80)
print("TRAINING COMPLETE! 🎉")
print("=" * 80)
print(f"\nYour model is ready at: {model_path}")
print("Model size: ~130MB")
print("Inference speed: ~5-10ms per headline on CPU")
print("\nYou can now classify millions of headlines efficiently!")
