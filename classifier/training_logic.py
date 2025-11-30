"""
Core training logic for SetFit water conflict classifier.

Provides training function that can be used by both local and HF Jobs scripts.
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitModelCardData


def train_model(train_dataset: Dataset,
                eval_dataset: Dataset,
                base_model: str = "BAAI/bge-small-en-v1.5",
                label_names: list[str] = None,
                batch_size: int = 16,
                num_epochs: int = 3,
                num_iterations: int = 20,
                sampling_strategy: str = "oversampling",
                model_card_data: SetFitModelCardData = None) -> SetFitModel:
    """
    Train SetFit multi-label classifier.
    
    Args:
        train_dataset: HF Dataset with 'text' and 'labels' columns
        eval_dataset: HF Dataset with 'text' and 'labels' columns
        base_model: Pretrained model to use as base
        label_names: List of label names (e.g. ['Trigger', 'Casualty', 'Weapon'])
        batch_size: Training batch size
        num_epochs: Number of training epochs
        num_iterations: Number of text pairs to generate for contrastive learning (default 20, reduce for larger datasets)
        sampling_strategy: 'oversampling' or 'undersampling'
        model_card_data: Optional SetFitModelCardData for model card metadata
        
    Returns:
        Trained SetFitModel
    """
    if label_names is None:
        label_names = ['Trigger', 'Casualty', 'Weapon']
    
    print(f"\n  Initializing model...")
    print(f"  Base model: {base_model}")
    print(f"  Strategy: One-vs-Rest multi-label classification")
    
    # Initialize model with balanced class weights for minority class handling
    model = SetFitModel.from_pretrained(
        base_model,
        multi_target_strategy="one-vs-rest",
        labels=label_names,
        model_card_data=model_card_data,
        head_params={"class_weight": "balanced"}  # Handle class imbalance (esp. Weapon)
    )
    print("  ✓ Model initialized")
    
    # Configure training
    print(f"\n  Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Iterations (contrastive pairs): {num_iterations}")
    print(f"  Sampling strategy: {sampling_strategy}")
    
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        body_learning_rate=2e-5,  # Slow fine-tuning of sentence transformer embeddings
        head_learning_rate=1e-2,   # Faster training for classification head
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        sampling_strategy=sampling_strategy,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="f1",
        column_mapping={"text": "text", "labels": "label"}
    )
    
    # Train
    print("\n  Starting training...")
    trainer.train()
    print("\n  ✓ Training complete!")
    
    return model

