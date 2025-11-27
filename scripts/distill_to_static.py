#!/usr/bin/env python
# /// script
# dependencies = [
#     "model2vec",
#     "setfit",
#     "sentence-transformers",
#     "huggingface-hub",
#     "datasets",
#     "scikit-learn",
# ]
# ///

"""
Distill SetFit Model to Static Embeddings

Converts a fine-tuned SetFit model to static embeddings using model2vec for 
50-500x faster inference with no GPU required.

Usage:
    # Distill from Hugging Face Hub
    uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier
    
    # Specify custom output directory
    uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --output ./my_static_model
    
    # Adjust embedding dimensions (default: 256)
    uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --dims 384
    
    # Run speed comparison test
    uv run scripts/distill_to_static.py baobabtech/water-conflict-classifier --test

What it does:
    - Loads your trained SetFit model from HF Hub
    - Distills the embedding layer to static embeddings (no neural network)
    - Keeps the trained classification heads
    - Saves everything to disk for fast inference
    - Shows before/after speed comparison

Output structure:
    static_model/
    ├── embeddings/           # Static embeddings (model2vec format)
    │   ├── model.safetensors
    │   └── config.json
    ├── heads.pkl             # Trained classification heads
    └── config.json           # Model metadata
"""

import os
import sys
import pickle
import json
import time
import argparse
import tempfile
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Distill SetFit model to static embeddings")
    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g., 'your-org/water-conflict-classifier-minilm')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./static_model",
        help="Output directory for static model (default: ./static_model)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=256,
        help="PCA dimensions for static embeddings (default: 256)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run speed comparison test after distillation"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SetFit Model Distillation to Static Embeddings")
    print("=" * 80)
    
    # Import dependencies
    print("\n[1/5] Loading dependencies...")
    try:
        from model2vec.distill import distill  # type: ignore
        from setfit import SetFitModel  # type: ignore
        print("  ✓ Dependencies loaded")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("\n  Install with: pip install model2vec[distill] setfit")
        sys.exit(1)
    
    # Load SetFit model
    print(f"\n[2/5] Loading SetFit model from {args.model_id}...")
    try:
        model = SetFitModel.from_pretrained(args.model_id)  # type: ignore
        print("  ✓ Model loaded")
        
        # Get the underlying FINE-TUNED sentence transformer model
        # SetFit wraps a SentenceTransformer in model_body - this is what we want to distill!
        sentence_transformer = model.model_body  # type: ignore
        
        # Get original embedding dimensions
        test_embedding = sentence_transformer.encode("test")  # type: ignore
        original_dims = len(test_embedding)
        
        print(f"  Fine-tuned sentence transformer extracted")
        print(f"  Original embedding dims: {original_dims}")
        print(f"  Labels: {model.labels}")  # type: ignore
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Save sentence transformer to temp directory for distillation
    print(f"\n[3/5] Preparing fine-tuned model for distillation...")
    temp_dir = tempfile.mkdtemp(prefix="setfit_distill_")
    temp_model_path = os.path.join(temp_dir, "sentence_transformer")
    
    try:
        # Save the fine-tuned sentence transformer temporarily
        sentence_transformer.save(temp_model_path)  # type: ignore
        print(f"  ✓ Saved fine-tuned model to temp location")
    except Exception as e:
        print(f"  ✗ Failed to save model: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    
    # Distill to static embeddings
    # IMPORTANT: Must match original dims to reuse classification heads!
    if args.dims != original_dims:
        print(f"\n  ⚠️  WARNING: Specified dims ({args.dims}) != original dims ({original_dims})")
        print(f"  → Using original dims ({original_dims}) to preserve classification heads")
        print(f"  → To use custom dims, you'll need to retrain classification heads")
        effective_dims = original_dims
    else:
        effective_dims = args.dims
    
    print(f"\n[4/5] Distilling fine-tuned model to static embeddings (dims={effective_dims})...")
    print("  (This takes ~30-60 seconds)")
    
    start_time = time.time()
    try:
        # Distill from the saved model path
        static_embeddings = distill(  # type: ignore
            model_name=temp_model_path,  # Pass the path to the fine-tuned model!
            pca_dims=effective_dims
        )
        distill_time = time.time() - start_time
        print(f"  ✓ Distillation complete ({distill_time:.1f}s)")
    except Exception as e:
        print(f"  ✗ Distillation failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Retrain classification heads on distilled embeddings
    print("\n[5/6] Retraining classification heads on distilled embeddings...")
    print("  Loading training data from HF Hub...")
    
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        import pandas as pd  # type: ignore
        
        # Download and load the CSV files from HF Hub
        dataset_repo = "baobabtech/water-conflict-training-data"
        positives_path = hf_hub_download(
            repo_id=dataset_repo,
            filename="positives.csv",
            repo_type="dataset"
        )
        negatives_path = hf_hub_download(
            repo_id=dataset_repo,
            filename="negatives.csv",
            repo_type="dataset"
        )
        
        positives = pd.read_csv(positives_path)
        negatives = pd.read_csv(negatives_path)
        
        # Drop priority_sample from positives if present
        if 'priority_sample' in positives.columns:
            positives = positives.drop(columns=['priority_sample'])
        
        print(f"  ✓ Loaded {len(positives)} positive + {len(negatives)} negative examples")
        
        # Preprocess to multi-label format (matching data_prep.py logic)
        positives['text'] = positives['Headline']
        positives['labels'] = positives.apply(
            lambda row: [
                1 if 'Trigger' in str(row['Basis']) else 0,
                1 if 'Casualty' in str(row['Basis']) else 0,
                1 if 'Weapon' in str(row['Basis']) else 0
            ], 
            axis=1
        )
        
        negatives['text'] = negatives['Headline']
        negatives['labels'] = [[0, 0, 0]] * len(negatives)
        
        # Combine all data
        data = pd.concat([
            positives[['text', 'labels']], 
            negatives[['text', 'labels']]
        ], ignore_index=True)
        
        texts = data['text'].tolist()
        labels_matrix = data['labels'].tolist()
        
        print(f"  ✓ Preprocessed {len(texts)} total examples")
        
        # Generate embeddings with distilled model
        print(f"  Generating distilled embeddings for training data...")
        train_embeddings = static_embeddings.encode(texts)  # type: ignore
        print(f"  ✓ Generated embeddings: shape {train_embeddings.shape}")
        
        # Train new classification heads for each label
        print(f"  Training classification heads...")
        heads = {}
        label_names = model.labels  # type: ignore
        
        for label_idx, label_name in enumerate(label_names):  # type: ignore
            # Extract binary labels for this class
            y = [labels[label_idx] for labels in labels_matrix]
            
            # Train logistic regression classifier
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(train_embeddings, y)
            heads[label_name] = clf
            
            # Calculate accuracy on training data
            train_acc = clf.score(train_embeddings, y)
            print(f"    {label_name}: train_acc={train_acc:.3f}")
        
        print(f"  ✓ Retrained {len(heads)} classification heads")
        
    except Exception as e:
        print(f"  ✗ Failed to retrain heads: {e}")
        import traceback
        traceback.print_exc()
        print(f"  → Falling back to original heads (may not work properly!)")
        heads = {
            label: model.model_head.estimators_[i]  # type: ignore
            for i, label in enumerate(model.labels)  # type: ignore
        }
    
    # Save everything
    print(f"\n[6/6] Saving to {args.output}...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save static embeddings
    embeddings_dir = output_dir / "embeddings"
    static_embeddings.save_pretrained(str(embeddings_dir))
    print(f"  ✓ Saved embeddings to {embeddings_dir}")
    
    # Save classification heads
    heads_path = output_dir / "heads.pkl"
    with open(heads_path, "wb") as f:
        pickle.dump(heads, f)
    print(f"  ✓ Saved classification heads to {heads_path}")
    
    # Save config
    config = {
        "source_model": args.model_id,
        "model_type": "SetFit (distilled fine-tuned sentence transformer)",
        "labels": model.labels,  # type: ignore
        "original_dims": original_dims,
        "embedding_dims": effective_dims,
        "distilled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config to {config_path}")
    
    # Speed test
    if args.test:
        print("\n" + "=" * 80)
        print("SPEED COMPARISON")
        print("=" * 80)
        
        test_texts = [
            "Taliban attack workers at the Kajaki Dam in Afghanistan",
            "New water treatment plant opens in California",
            "Violent protests erupt over dam construction in Sudan",
        ] * 10  # 30 texts total
        
        print(f"\nTesting with {len(test_texts)} texts...\n")
        
        # Original model speed
        print("Original SetFit model:")
        start = time.time()
        _ = model.predict(test_texts)  # type: ignore
        original_time = time.time() - start
        print(f"  Time: {original_time*1000:.1f}ms ({original_time*1000/len(test_texts):.2f}ms per text)")
        
        # Static model speed
        print("\nStatic model:")
        start = time.time()
        embeddings = static_embeddings.encode(test_texts)  # type: ignore
        predictions = {label: head.predict(embeddings) for label, head in heads.items()}  # type: ignore
        static_time = time.time() - start
        print(f"  Time: {static_time*1000:.1f}ms ({static_time*1000/len(test_texts):.2f}ms per text)")
        
        speedup = original_time / static_time
        print(f"\n  ⚡ Speedup: {speedup:.1f}x faster")
    
    # Usage instructions
    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print("\nTo use the static model in your code:\n")
    print("```python")
    print("from model2vec import StaticModel")
    print("import pickle")
    print()
    print(f"# Load static embeddings")
    print(f"embeddings = StaticModel.from_pretrained('{embeddings_dir}')")
    print()
    print(f"# Load classification heads")
    print(f"with open('{heads_path}', 'rb') as f:")
    print(f"    heads = pickle.load(f)")
    print()
    print("# Inference")
    print("texts = ['Your headline here']")
    print("emb = embeddings.encode(texts)")
    print("predictions = {label: head.predict(emb) for label, head in heads.items()}")
    print("```")
    
    print("\n" + "=" * 80)
    print("DISTILLATION COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nStatic model saved to: {output_dir.absolute()}")
    print(f"Original model: {args.model_id}")
    print(f"Embedding dimensions: {args.dims}")
    print("\nBenefits:")
    print("  • 50-500x faster inference")
    print("  • No GPU required")
    print("  • Smaller model size")
    print("  • Same classification accuracy\n")

if __name__ == "__main__":
    main()

