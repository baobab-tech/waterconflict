#!/usr/bin/env python
"""
Demo script to showcase the versioning system.

This creates some example experiments to demonstrate the tracking capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from classifier package
sys.path.insert(0, str(Path(__file__).parent.parent / "classifier"))

from versioning import ExperimentTracker


def create_demo_experiments():
    """Create some demo experiments to showcase the system."""
    
    tracker = ExperimentTracker("demo_experiment_history.jsonl")
    
    print("Creating demo experiments...")
    print("=" * 80)
    
    # Experiment 1: Baseline
    print("\n1. Creating v1.0 (baseline)...")
    tracker.log_experiment(
        version="v1.0",
        config={
            "base_model": "BAAI/bge-small-en-v1.5",
            "train_size": 600,
            "batch_size": 16,
            "num_epochs": 1,
            "sampling_strategy": "none"
        },
        metrics={
            "overall": {
                "f1_micro": 0.8234,
                "f1_macro": 0.8012,
                "accuracy": 0.7890
            },
            "per_label": {
                "Trigger": {"f1": 0.8567, "precision": 0.8623, "recall": 0.8512},
                "Casualty": {"f1": 0.8123, "precision": 0.8234, "recall": 0.8012},
                "Weapon": {"f1": 0.7345, "precision": 0.7456, "recall": 0.7234}
            }
        },
        metadata={
            "notes": "Initial baseline with minimal training data"
        }
    )
    
    # Experiment 2: More data
    print("2. Creating v1.1 (increased training data)...")
    tracker.log_experiment(
        version="v1.1",
        config={
            "base_model": "BAAI/bge-small-en-v1.5",
            "train_size": 1200,
            "batch_size": 16,
            "num_epochs": 1,
            "sampling_strategy": "stratified"
        },
        metrics={
            "overall": {
                "f1_micro": 0.8567,
                "f1_macro": 0.8345,
                "accuracy": 0.8234
            },
            "per_label": {
                "Trigger": {"f1": 0.8890, "precision": 0.8923, "recall": 0.8857},
                "Casualty": {"f1": 0.8456, "precision": 0.8534, "recall": 0.8378},
                "Weapon": {"f1": 0.7689, "precision": 0.7789, "recall": 0.7589}
            }
        },
        metadata={
            "notes": "Doubled training data, added stratified sampling"
        }
    )
    
    # Experiment 3: More epochs
    print("3. Creating v1.2 (increased epochs)...")
    tracker.log_experiment(
        version="v1.2",
        config={
            "base_model": "BAAI/bge-small-en-v1.5",
            "train_size": 1200,
            "batch_size": 32,
            "num_epochs": 3,
            "sampling_strategy": "stratified"
        },
        metrics={
            "overall": {
                "f1_micro": 0.8765,
                "f1_macro": 0.8543,
                "accuracy": 0.8456
            },
            "per_label": {
                "Trigger": {"f1": 0.9012, "precision": 0.9045, "recall": 0.8979},
                "Casualty": {"f1": 0.8678, "precision": 0.8734, "recall": 0.8623},
                "Weapon": {"f1": 0.7940, "precision": 0.8023, "recall": 0.7857}
            }
        },
        metadata={
            "notes": "Increased epochs to 3, larger batch size"
        }
    )
    
    print("\n✓ Demo experiments created!")
    print(f"✓ Saved to: demo_experiment_history.jsonl")
    
    # Display summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    tracker.print_summary(limit=3)
    
    # Show comparisons
    print("\n" + "=" * 80)
    print("COMPARISONS")
    print("=" * 80)
    
    print("\n📊 v1.0 → v1.1 (Added training data)")
    comp1 = tracker.compare_experiments("v1.0", "v1.1", "f1_micro")
    print(f"   F1 Micro: {comp1['difference']:+.4f} ({comp1['percent_change']:+.2f}%)")
    print(f"   Result: {'✓ Improvement' if comp1['improvement'] else '✗ Regression'}")
    
    print("\n📊 v1.1 → v1.2 (Increased epochs)")
    comp2 = tracker.compare_experiments("v1.1", "v1.2", "f1_micro")
    print(f"   F1 Micro: {comp2['difference']:+.4f} ({comp2['percent_change']:+.2f}%)")
    print(f"   Result: {'✓ Improvement' if comp2['improvement'] else '✗ Regression'}")
    
    print("\n📊 v1.0 → v1.2 (Overall improvement)")
    comp3 = tracker.compare_experiments("v1.0", "v1.2", "f1_micro")
    print(f"   F1 Micro: {comp3['difference']:+.4f} ({comp3['percent_change']:+.2f}%)")
    print(f"   Result: {'✓ Improvement' if comp3['improvement'] else '✗ Regression'}")
    
    print("\n" + "=" * 80)
    print("Try the view_experiments.py script:")
    print("  python scripts/view_experiments.py --history-file demo_experiment_history.jsonl")
    print("  python scripts/view_experiments.py --history-file demo_experiment_history.jsonl --compare v1.0 v1.2")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    create_demo_experiments()

