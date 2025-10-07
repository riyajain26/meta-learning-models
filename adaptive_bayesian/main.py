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

def main(args):
    config = load_config(args.config)
    set_seed(config.get("random_seed", 42))

    # Select device: use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy dataset for testing
    print("\nCreating dummy dataset...")
    n_subjects = 15
    samples_per_subject = 100
    n_total = n_subjects * samples_per_subject
    
    dummy_data = np.random.randn(n_total, 32, 1044)
    dummy_labels = np.random.randint(0, 4, n_total)
    dummy_subjects = np.repeat(np.arange(n_subjects), samples_per_subject)
    
    dataset = EEGDataset(dummy_data, dummy_labels, dummy_subjects)
    print(f"Dataset created: {len(dataset)} samples")
    
    # Create model
    print("\nCreating ABML model...")
    model = ABML(
        in_channels=32,
        time_length=1044,
        num_classes=4,
        latent_dim=64,
        hidden_dim=128,
        meta_param_rank=32,
        temperature_init=1.0,
        temperature_min=0.1
    ).to(device)

    # Quick test with one subject
    print("\n" + "="*60)
    print("Quick Test: Training on subjects 0-13, testing on subject 14")
    print("="*60)
    
    train_data, train_labels = dataset.get_subjects_data(list(range(14)))
    test_data, test_labels = dataset.get_subject_data(14)
    
    trainer = ABMLTrainer(model, device, learning_rate=1e-3, alpha=1.0, beta=0.1)
    
    history = trainer.train(
        train_data, train_labels,
        test_data, test_labels,
        n_epochs=10,
        support_pool_size=50,
        batch_size=4,
        eval_every=5,
        save_path='abml_test.pth'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABML with EEG Data using YAML Config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)