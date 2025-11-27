"""
Model versioning and experiment tracking for water conflict classifier.

Provides utilities for:
- Logging experiment metadata to JSON
- Creating version tags on HF Hub
- Comparing experiments across versions
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExperimentTracker:
    """Track experiments and model versions."""
    
    def __init__(self, history_file: str = "experiment_history.jsonl"):
        """
        Initialize experiment tracker.
        
        Args:
            history_file: Path to JSONL file storing experiment history
        """
        self.history_file = Path(history_file)
        
    def log_experiment(self,
                      version: str,
                      config: dict,
                      metrics: dict,
                      metadata: Optional[dict] = None) -> dict:
        """
        Log an experiment to the history file.
        
        Args:
            version: Version identifier (e.g., "v1.0", "v1.1")
            config: Training configuration (model, hyperparams, data settings)
            metrics: Evaluation metrics (F1 scores, accuracy, etc.)
            metadata: Optional additional metadata (git commit, notes, etc.)
            
        Returns:
            Dictionary with complete experiment record
        """
        # Clean metrics - remove numpy arrays and convert numpy types to Python types
        cleaned_metrics = self._clean_metrics_for_json(metrics)
        
        experiment = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": cleaned_metrics,
            "metadata": metadata or {}
        }
        
        # Append to history file (JSONL format)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(experiment) + "\n")
        
        return experiment
    
    def _clean_metrics_for_json(self, metrics: dict) -> dict:
        """
        Clean metrics dict for JSON serialization.
        Removes numpy arrays and converts numpy types to Python types.
        """
        import numpy as np
        
        cleaned = {}
        for key, value in metrics.items():
            # Skip numpy arrays (y_true, y_pred)
            if isinstance(value, np.ndarray):
                continue
            # Recursively clean nested dicts
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metrics_for_json(value)
            # Convert numpy types to Python types
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        
        return cleaned
    
    def get_experiments(self, limit: Optional[int] = None) -> list[dict]:
        """
        Load experiment history from file.
        
        Args:
            limit: Optional limit on number of experiments to return (most recent)
            
        Returns:
            List of experiment dictionaries
        """
        if not self.history_file.exists():
            return []
        
        experiments = []
        with open(self.history_file, "r") as f:
            for line in f:
                if line.strip():
                    experiments.append(json.loads(line))
        
        if limit:
            experiments = experiments[-limit:]
        
        return experiments
    
    def get_experiment_by_version(self, version: str) -> Optional[dict]:
        """Get a specific experiment by version identifier."""
        experiments = self.get_experiments()
        for exp in reversed(experiments):  # Search from most recent
            if exp["version"] == version:
                return exp
        return None
    
    def compare_experiments(self, 
                           version1: str, 
                           version2: str,
                           metric_key: str = "f1_micro") -> dict:
        """
        Compare two experiments.
        
        Args:
            version1: First version to compare
            version2: Second version to compare
            metric_key: Metric to compare (default: f1_micro)
            
        Returns:
            Dictionary with comparison results
        """
        exp1 = self.get_experiment_by_version(version1)
        exp2 = self.get_experiment_by_version(version2)
        
        if not exp1 or not exp2:
            return {"error": "One or both versions not found"}
        
        # Extract metric from nested structure
        def get_metric(exp, key):
            if key in exp['metrics'].get('overall', {}):
                return exp['metrics']['overall'][key]
            return exp['metrics'].get(key)
        
        metric1 = get_metric(exp1, metric_key)
        metric2 = get_metric(exp2, metric_key)
        
        if metric1 is None or metric2 is None:
            return {"error": f"Metric '{metric_key}' not found in experiments"}
        
        diff = metric2 - metric1
        pct_change = (diff / metric1) * 100 if metric1 != 0 else 0
        
        return {
            "version1": {"version": version1, metric_key: metric1},
            "version2": {"version": version2, metric_key: metric2},
            "difference": diff,
            "percent_change": pct_change,
            "improvement": diff > 0
        }
    
    def print_summary(self, limit: int = 5):
        """Print a summary of recent experiments."""
        experiments = self.get_experiments(limit=limit)
        
        if not experiments:
            print("No experiments logged yet.")
            return
        
        print("\n" + "=" * 80)
        print(f"EXPERIMENT HISTORY (last {min(limit, len(experiments))})")
        print("=" * 80)
        
        for exp in reversed(experiments):  # Most recent first
            print(f"\n{exp['version']} - {exp['timestamp'][:19]}")
            
            # Config summary
            config = exp['config']
            print(f"  Config:")
            if 'train_size' in config:
                print(f"    Training samples: {config['train_size']}")
            if 'base_model' in config:
                print(f"    Base model: {config['base_model']}")
            if 'num_epochs' in config:
                print(f"    Epochs: {config['num_epochs']}")
            
            # Metrics summary
            metrics = exp['metrics']
            print(f"  Metrics:")
            if 'overall' in metrics:
                overall = metrics['overall']
                print(f"    F1 (micro): {overall.get('f1_micro', 'N/A'):.4f}")
                print(f"    Accuracy: {overall.get('accuracy', 'N/A'):.4f}")
            
            if 'per_label' in metrics:
                per_label = metrics['per_label']
                for label, vals in per_label.items():
                    print(f"    {label} F1: {vals.get('f1', 'N/A'):.4f}")


def create_hf_version_tag(model_repo: str,
                         version: str,
                         metrics: dict,
                         config: dict,
                         token: Optional[str] = None) -> bool:
    """
    Create a git tag on HuggingFace Hub repository.
    
    Args:
        model_repo: Full repo ID (org/repo-name)
        version: Version identifier (e.g., "v1.0")
        metrics: Evaluation metrics to include in tag message
        config: Training config to include in tag message
        token: Optional HF token (uses default if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi, create_tag
        
        api = HfApi(token=token)
        
        # Check if tag already exists
        try:
            refs = api.list_repo_refs(repo_id=model_repo, repo_type="model")
            existing_tags = [tag.ref for tag in refs.tags] if hasattr(refs, 'tags') else []
            if f"refs/tags/{version}" in existing_tags or version in existing_tags:
                print(f"  ℹ Tag {version} already exists, skipping creation")
                return True
        except Exception:
            # If we can't check, try to create anyway
            pass
        
        # Build tag message with key metrics
        tag_message_parts = [f"Version {version}"]
        
        # Add metrics
        if 'overall' in metrics:
            overall = metrics['overall']
            tag_message_parts.append(
                f"F1={overall.get('f1_micro', 0):.4f}"
            )
            tag_message_parts.append(
                f"Acc={overall.get('accuracy', 0):.4f}"
            )
        
        # Add key config items
        if 'train_size' in config:
            tag_message_parts.append(f"samples={config['train_size']}")
        
        tag_message = " | ".join(tag_message_parts)
        
        # Create the tag
        create_tag(
            repo_id=model_repo,
            tag=version,
            tag_message=tag_message,
            repo_type="model",
            token=token
        )
        
        print(f"  ✓ Created HF Hub tag: {version}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        # Handle case where tag already exists
        if "409" in error_msg or "already exists" in error_msg.lower():
            print(f"  ℹ Tag {version} already exists, skipping creation")
            return True
        print(f"  ✗ Failed to create HF Hub tag: {e}")
        return False


def get_next_version(history_file: str = "experiment_history.jsonl",
                    major: bool = False,
                    minor: bool = False) -> str:
    """
    Get next version number based on experiment history.
    
    Args:
        history_file: Path to experiment history file
        major: If True, increment major version (X.0)
        minor: If True, increment minor version (X.Y)
        
    Returns:
        Next version string (e.g., "v1.1")
    """
    tracker = ExperimentTracker(history_file)
    experiments = tracker.get_experiments()
    
    if not experiments:
        return "v1.0"
    
    # Get last version
    last_version = experiments[-1]['version']
    
    # Parse version (expecting format like "v1.0" or "v1.1")
    try:
        parts = last_version.lstrip('v').split('.')
        major_num = int(parts[0])
        minor_num = int(parts[1]) if len(parts) > 1 else 0
        
        if major:
            return f"v{major_num + 1}.0"
        elif minor:
            return f"v{major_num}.{minor_num + 1}"
        else:
            # Default: increment minor
            return f"v{major_num}.{minor_num + 1}"
    except:
        # If parsing fails, just increment
        return "v1.0"

