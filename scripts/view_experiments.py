#!/usr/bin/env python
"""
View and compare experiment history.

Usage:
    # View recent experiments
    python view_experiments.py
    
    # View specific number of experiments
    python view_experiments.py --limit 10
    
    # Compare two versions
    python view_experiments.py --compare v1.0 v1.1
    
    # Show all experiments
    python view_experiments.py --all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from classifier package
sys.path.insert(0, str(Path(__file__).parent.parent / "classifier"))

from versioning import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(
        description="View and compare model experiment history"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="Number of recent experiments to show (default: 5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all experiments"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("VERSION1", "VERSION2"),
        help="Compare two versions (e.g., --compare v1.0 v1.1)"
    )
    parser.add_argument(
        "--metric",
        default="f1_micro",
        help="Metric to use for comparison (default: f1_micro)"
    )
    parser.add_argument(
        "--history-file",
        default="experiment_history.jsonl",
        help="Path to experiment history file"
    )
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ExperimentTracker(args.history_file)
    
    # Compare mode
    if args.compare:
        version1, version2 = args.compare
        print(f"\nComparing {version1} vs {version2}")
        print("=" * 80)
        
        comparison = tracker.compare_experiments(version1, version2, args.metric)
        
        if "error" in comparison:
            print(f"\n✗ {comparison['error']}")
            return
        
        v1_info = comparison['version1']
        v2_info = comparison['version2']
        diff = comparison['difference']
        pct = comparison['percent_change']
        improved = comparison['improvement']
        
        print(f"\n{v1_info['version']}: {args.metric} = {v1_info[args.metric]:.4f}")
        print(f"{v2_info['version']}: {args.metric} = {v2_info[args.metric]:.4f}")
        print(f"\nDifference: {diff:+.4f} ({pct:+.2f}%)")
        
        if improved:
            print("✓ Improvement detected")
        else:
            print("✗ Performance decreased")
        
        return
    
    # Summary mode
    limit = None if args.all else args.limit
    tracker.print_summary(limit=limit if limit else 999)
    
    # Show available versions for comparison
    experiments = tracker.get_experiments()
    if len(experiments) >= 2:
        print("\n" + "=" * 80)
        print("COMPARE EXPERIMENTS")
        print("=" * 80)
        versions = [exp['version'] for exp in experiments]
        print("\nAvailable versions:", ", ".join(versions))
        print(f"\nExample comparison:")
        print(f"  python {Path(__file__).name} --compare {versions[-2]} {versions[-1]}")


if __name__ == "__main__":
    main()

