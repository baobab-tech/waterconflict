#!/usr/bin/env python
"""
View and Compare Evaluation Results from HuggingFace

This script loads the evaluation results dataset from HF Hub and provides
various views for comparing experiments.

Usage:
    # View all experiments (sorted by F1)
    python view_evals.py
    
    # Show top 10 experiments
    python view_evals.py --top 10
    
    # Compare two specific versions
    python view_evals.py --compare v1.0 v1.1
    
    # Analyze impact of hyperparameters
    python view_evals.py --analyze sample_size
    python view_evals.py --analyze num_epochs
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import HF_ORGANIZATION
except ImportError:
    print("\n✗ config.py not found!")
    print("  Copy config.sample.py to config.py\n")
    sys.exit(1)

EVALS_REPO = f"{HF_ORGANIZATION}/water-conflict-classifier-evals"


def load_evals() -> pd.DataFrame:
    """Load evaluation results from HF Hub."""
    try:
        from datasets import load_dataset
        ds = load_dataset(EVALS_REPO, split="train")
        return ds.to_pandas()
    except Exception as e:
        print(f"✗ Error loading evals dataset: {e}")
        print(f"  Repo: {EVALS_REPO}")
        print(f"  Make sure the dataset exists and you're authenticated")
        sys.exit(1)


def show_all_experiments(df: pd.DataFrame, top_n: int = None):
    """Display all experiments sorted by F1 score."""
    print("=" * 100)
    print(f"ALL EXPERIMENTS (sorted by F1 micro)")
    print("=" * 100)
    
    # Select key columns
    columns = [
        'version', 'timestamp', 'f1_micro', 'f1_macro', 'accuracy',
        'train_size', 'sample_size', 'num_epochs', 'batch_size', 'num_iterations'
    ]
    
    # Filter columns that exist
    available_cols = [col for col in columns if col in df.columns]
    display_df = df[available_cols].copy()
    
    # Sort by F1 micro
    if 'f1_micro' in display_df.columns:
        display_df = display_df.sort_values('f1_micro', ascending=False)
    
    # Limit if requested
    if top_n:
        display_df = display_df.head(top_n)
        print(f"\nShowing top {top_n} experiments:")
    else:
        print(f"\nShowing all {len(display_df)} experiments:")
    
    # Format display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Shorten timestamp for display
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].str[:19]
    
    print("\n" + display_df.to_string(index=False))
    print("\n")


def compare_versions(df: pd.DataFrame, version1: str, version2: str):
    """Compare two specific versions."""
    print("=" * 100)
    print(f"COMPARING: {version1} vs {version2}")
    print("=" * 100)
    
    # Get experiments
    exp1 = df[df['version'] == version1]
    exp2 = df[df['version'] == version2]
    
    if exp1.empty:
        print(f"\n✗ Version {version1} not found!")
        return
    if exp2.empty:
        print(f"\n✗ Version {version2} not found!")
        return
    
    exp1 = exp1.iloc[0]
    exp2 = exp2.iloc[0]
    
    # Compare metrics
    metrics = ['f1_micro', 'f1_macro', 'accuracy', 'hamming_loss']
    
    print(f"\nMetrics Comparison:")
    print(f"\n{'Metric':<20} {version1:<15} {version2:<15} {'Diff':<15} {'% Change':<15}")
    print("-" * 80)
    
    for metric in metrics:
        if metric in exp1 and metric in exp2:
            v1 = exp1[metric]
            v2 = exp2[metric]
            diff = v2 - v1
            pct_change = (diff / v1) * 100 if v1 != 0 else 0
            
            # Color code improvement/degradation
            indicator = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            
            print(f"{metric:<20} {v1:<15.4f} {v2:<15.4f} {diff:<15.4f} {pct_change:<14.2f}% {indicator}")
    
    # Compare configuration
    config_fields = ['train_size', 'sample_size', 'batch_size', 'num_epochs', 'num_iterations']
    
    print(f"\n\nConfiguration Comparison:")
    print(f"\n{'Config':<20} {version1:<15} {version2:<15}")
    print("-" * 50)
    
    for field in config_fields:
        if field in exp1 and field in exp2:
            v1 = exp1[field]
            v2 = exp2[field]
            print(f"{field:<20} {str(v1):<15} {str(v2):<15}")
    
    print("\n")


def analyze_hyperparameter(df: pd.DataFrame, param: str):
    """Analyze impact of a hyperparameter on performance."""
    print("=" * 100)
    print(f"ANALYZING: Impact of {param} on Performance")
    print("=" * 100)
    
    if param not in df.columns:
        print(f"\n✗ Parameter '{param}' not found in dataset!")
        print(f"  Available parameters: {', '.join(df.columns)}")
        return
    
    # Group by parameter and calculate mean metrics
    grouped = df.groupby(param).agg({
        'f1_micro': ['mean', 'std', 'count'],
        'f1_macro': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    print(f"\n{grouped.to_string()}")
    
    # Show best value
    best_idx = grouped[('f1_micro', 'mean')].idxmax()
    best_value = grouped.loc[best_idx, ('f1_micro', 'mean')]
    
    print(f"\n✓ Best {param}: {best_idx} (F1 micro: {best_value:.4f})")
    print("\n")


def show_per_label_metrics(df: pd.DataFrame, top_n: int = 5):
    """Show per-label metrics for top experiments."""
    print("=" * 100)
    print(f"PER-LABEL METRICS (Top {top_n} by F1 micro)")
    print("=" * 100)
    
    # Sort by F1 and take top N
    if 'f1_micro' in df.columns:
        df_sorted = df.sort_values('f1_micro', ascending=False).head(top_n)
    else:
        df_sorted = df.head(top_n)
    
    for idx, row in df_sorted.iterrows():
        print(f"\n{row['version']} - F1 Micro: {row.get('f1_micro', 'N/A'):.4f}")
        print("-" * 80)
        
        # Show per-label metrics
        labels = ['trigger', 'casualty', 'weapon']
        print(f"{'Label':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        print("-" * 60)
        
        for label in labels:
            precision = row.get(f'{label}_precision', None)
            recall = row.get(f'{label}_recall', None)
            f1 = row.get(f'{label}_f1', None)
            support = row.get(f'{label}_support', None)
            
            if precision is not None:
                print(f"{label.capitalize():<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12.0f}")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="View and compare evaluation results from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--top',
        type=int,
        metavar='N',
        help='Show only top N experiments'
    )
    
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('V1', 'V2'),
        help='Compare two versions (e.g., --compare v1.0 v1.1)'
    )
    
    parser.add_argument(
        '--analyze',
        type=str,
        metavar='PARAM',
        help='Analyze impact of hyperparameter (e.g., --analyze sample_size)'
    )
    
    parser.add_argument(
        '--labels',
        action='store_true',
        help='Show detailed per-label metrics'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading evaluation results from: {EVALS_REPO}")
    df = load_evals()
    print(f"✓ Loaded {len(df)} experiments\n")
    
    # Execute requested action
    if args.compare:
        compare_versions(df, args.compare[0], args.compare[1])
    elif args.analyze:
        analyze_hyperparameter(df, args.analyze)
    elif args.labels:
        show_per_label_metrics(df, args.top or 5)
    else:
        show_all_experiments(df, args.top)


if __name__ == "__main__":
    main()

