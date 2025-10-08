import torch
import pandas as pd
import argparse
import yaml
from sklearn.model_selection import train_test_split
import numpy as np

from adaptive_bayesian.core.abml import ABML
from adaptive_bayesian.core.abml_trainer import ABMLTrainer
from adaptive_bayesian.core.eeg_meta_dataset import EEGDataset
from adaptive_bayesian.utils.utils import set_seed

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dummy_dataset(n_subjects=15, samples_per_subject=100, 
                        in_channels=32, time_length=1044, num_classes=4):
    """
    Create a dummy EEG dataset for testing
    
    Args:
        n_subjects: number of subjects
        samples_per_subject: samples per subject
        in_channels: number of EEG channels
        time_length: length of time series
        num_classes: number of classes
        
    Returns:
        dataset: EEGDataset object
    """
    n_total = n_subjects * samples_per_subject
    
    # Generate synthetic EEG-like data
    dummy_data = np.random.randn(n_total, in_channels, time_length).astype(np.float32)
    
    # Add some class-specific patterns to make it learnable
    dummy_labels = np.random.randint(0, num_classes, n_total)
    for i, label in enumerate(dummy_labels):
        # Add small class-specific pattern
        dummy_data[i] += 0.1 * label * np.random.randn(in_channels, time_length)
    
    dummy_subjects = np.repeat(np.arange(n_subjects), samples_per_subject)
    
    dataset = EEGDataset(dummy_data, dummy_labels, dummy_subjects)
    return dataset


def main(args):
    config = load_config(args.config)
    set_seed(config.get("random_seed", 42))

    # Select device: use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy dataset for testing
    print("\nCreating dummy dataset...")
    dataset = create_dummy_dataset(
        n_subjects=config.get('n_subjects', 15),
        samples_per_subject=config.get('samples_per_subject', 100),
        in_channels=config.get('in_channels', 32),
        time_length=config.get('time_length', 1044),
        num_classes=config.get('num_classes', 4)
    )
 
    print(f"Dataset created: {len(dataset)} samples")

    # Split data: train on subjects 0 to n-2, test on subject n-1
    n_subjects = config.get('n_subjects', 15)
    train_subjects = list(range(n_subjects - 1))
    test_subject = n_subjects - 1
    
    train_data, train_labels = dataset.get_subjects_data(train_subjects)
    test_data, test_labels = dataset.get_subject_data(test_subject)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create model
    print("\nCreating ABML model...")
    model = ABML(
        in_channels=config.get('in_channels', 32),
        time_length=config.get('time_length', 1044),
        num_classes=config.get('num_classes', 4),
        latent_dim=config.get('latent_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        meta_param_rank=config.get('meta_param_rank', 32),
        temperature_init=config.get('temperature_init', 1.0),
        temperature_min=config.get('temperature_min', 0.1)
    ).to(device)
    
    # Create trainer
    trainer = ABMLTrainer(
        model, 
        device, 
        learning_rate=config.get('learning_rate', 1e-3),
        alpha=config.get('alpha', 1.0),
        beta=config.get('beta', 0.1)
    )
    
    # Train
    history = trainer.train(
        train_data, train_labels,
        test_data, test_labels,
        n_epochs=config.get('n_epochs', 100),
        support_pool_size=config.get('support_pool_size', 100),
        batch_size=config.get('batch_size', 8),
        eval_every=config.get('eval_every', 5),
        save_path=config.get('save_path', 'abml_best.pth')
    )

    print("\nTraining completed successfully!")
    print(f"Final best accuracy: {history['best_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABML with EEG Data using YAML Config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)