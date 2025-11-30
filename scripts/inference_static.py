#!/usr/bin/env python
# /// script
# dependencies = [
#     "model2vec",
#     "scikit-learn",
# ]
# ///

"""
Inference with Static Water Conflict Classifier

Fast inference using distilled static model (50-500x faster than SetFit).

Usage:
    # Predict from command line
    uv run scripts/inference_static.py "Military group attack workers at dam"
    
    # Predict multiple texts
    uv run scripts/inference_static.py "Text 1" "Text 2" "Text 3"
    
    # Specify model path
    uv run scripts/inference_static.py "Text" --model ./static_model

Example:
    $ uv run scripts/inference_static.py \\
        "Military group attack workers at the Kajaki Dam" \\
        "New water treatment plant opens in California"
    
    Text: Military group attacks workers at the Kajaki Dam
    Labels: Trigger, Casualty, Weapon
    
    Text: New water treatment plant opens in California
    Labels: None (not a water conflict)
"""

import argparse
import pickle
import sys
from pathlib import Path

def load_static_model(model_path):
    """Load static embeddings and classification heads."""
    try:
        from model2vec import StaticModel
    except ImportError:
        print("Error: model2vec not installed")
        print("Install with: pip install model2vec")
        sys.exit(1)
    
    model_path = Path(model_path)
    
    # Load static embeddings
    embeddings_path = model_path / "embeddings"
    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        print("Run distill_to_static.py first to create the static model")
        sys.exit(1)
    
    embeddings = StaticModel.from_pretrained(str(embeddings_path))
    
    # Load classification heads
    heads_path = model_path / "heads.pkl"
    if not heads_path.exists():
        print(f"Error: Classification heads not found at {heads_path}")
        sys.exit(1)
    
    with open(heads_path, "rb") as f:
        heads = pickle.load(f)
    
    return embeddings, heads

def predict(texts, embeddings, heads, debug=False):
    """Run inference on texts."""
    # Encode texts to embeddings
    emb = embeddings.encode(texts)
    
    if debug:
        print(f"\nDEBUG: Embedding shape: {emb.shape}")
        print(f"DEBUG: First embedding (first 10 dims): {emb[0][:10]}")
    
    # Get predictions from each classification head
    results = []
    for i, text in enumerate(texts):
        labels_detected = []
        if debug:
            print(f"\nDEBUG: Text {i}: '{text[:50]}...'")
        
        for label, head in heads.items():
            pred = head.predict([emb[i]])[0]
            
            # Get decision scores for debugging
            if debug and hasattr(head, 'decision_function'):
                score = head.decision_function([emb[i]])[0]
                print(f"  {label}: pred={pred}, score={score:.4f}")
            
            if pred == 1:
                labels_detected.append(label)
        results.append(labels_detected)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Predict water conflict labels using static model"
    )
    parser.add_argument(
        "texts",
        nargs="+",
        type=str,
        help="Text(s) to classify"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./static_model",
        help="Path to static model directory (default: ./static_model)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading static model from {args.model}...")
    embeddings, heads = load_static_model(args.model)
    print(f"✓ Model loaded ({len(heads)} labels)")
    
    # Predict
    print(f"\nPredicting {len(args.texts)} text(s)...\n")
    predictions = predict(args.texts, embeddings, heads, debug=args.debug)
    
    # Display results
    for text, labels in zip(args.texts, predictions):
        print(f"Text: {text}")
        if labels:
            print(f"Labels: {', '.join(labels)}")
        else:
            print("Labels: None (not a water conflict)")
        print()

if __name__ == "__main__":
    main()

