"""
Model evaluation and metrics for water conflict classifier.

Provides functions for evaluating trained models and formatting results.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from setfit import SetFitModel


def evaluate_model(model: SetFitModel, 
                   test_texts: list[str], 
                   test_labels: list[list[int]],
                   label_names: list[str] = None) -> dict:
    """
    Evaluate model on test data and return comprehensive metrics.
    
    Args:
        model: Trained SetFitModel
        test_texts: List of test headline strings
        test_labels: List of label arrays [[1,0,0], [0,1,1], ...]
        label_names: List of label names for per-label metrics
        
    Returns:
        Dictionary with overall metrics and per-label metrics
    """
    if label_names is None:
        label_names = ['Trigger', 'Casualty', 'Weapon']
    
    # Get predictions
    predictions = model.predict(test_texts)
    y_true = np.array(test_labels)
    y_pred = np.array(predictions)
    
    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_samples': f1_score(y_true, y_pred, average='samples')
    }
    
    # Per-label metrics
    per_label_metrics = {}
    for i, label_name in enumerate(label_names):
        precision = (y_pred[:, i] & y_true[:, i]).sum() / (y_pred[:, i].sum() + 1e-10)
        recall = (y_pred[:, i] & y_true[:, i]).sum() / (y_true[:, i].sum() + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        support = int(y_true[:, i].sum())
        
        per_label_metrics[label_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return {
        'overall': overall_metrics,
        'per_label': per_label_metrics,
        'y_true': y_true,
        'y_pred': y_pred
    }


def print_evaluation_results(results: dict, label_names: list[str] = None):
    """
    Pretty print evaluation results to console.
    
    Args:
        results: Dictionary returned from evaluate_model()
        label_names: List of label names
    """
    if label_names is None:
        label_names = ['Trigger', 'Casualty', 'Weapon']
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall metrics
    overall = results['overall']
    print(f"\nOverall Metrics:")
    print(f"  Accuracy (exact match): {overall['accuracy']:.4f}")
    print(f"  Hamming Loss:          {overall['hamming_loss']:.4f}")
    print(f"  F1 (micro):            {overall['f1_micro']:.4f}")
    print(f"  F1 (macro):            {overall['f1_macro']:.4f}")
    print(f"  F1 (samples):          {overall['f1_samples']:.4f}")
    
    # Per-label metrics
    print(f"\nPer-Label Metrics:")
    for label_name in label_names:
        metrics = results['per_label'][label_name]
        print(f"\n  {label_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Support:   {metrics['support']}")

