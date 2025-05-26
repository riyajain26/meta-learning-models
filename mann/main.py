import torch
import pandas as pd
import argparse
import yaml
from sklearn.model_selection import train_test_split

from mann.core.eeg_encoder import EEGEmbeddingNet
from mann.core.eeg_meta_dataset import EEGMetaDataset
from mann.core.mann_model import MANN, MANNWrapper
from mann.core.train import mann_train
from mann.core.test import mann_test
from mann.utils.utils import set_seed


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def load_data(metadata_path, test_size=0.2, random_state=42):
    # Load metadata CSV containing EEG recordings info (participant, labels, file paths, etc.)
    df = pd.read_csv(metadata_path)

    # Get unique participants and split into train/test groups (80% train, 20% test)
    participants = df['participant'].unique()
    train_participants, test_participants = train_test_split(participants, test_size=test_size, random_state=random_state)

    # Filter the DataFrame to create train and test DataFrames based on participant splits
    train_df = df[df['participant'].isin(train_participants)]
    test_df = df[df['participant'].isin(test_participants)]

    return train_df, test_df


def main(args):
    config = load_config(args.config)
    mode = args.mode

    set_seed(config.get("random_seed", 42))

     # Select device: use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df = load_data(config['metadata_csv'], test_size=config['test_size'])
    
    # EEG encoder
    eeg_encoder = EEGEmbeddingNet(
        embedding_dim=config["embedding_dim"],
        in_channels=config["in_channels"],
        input_time=config["input_time"]
    )

    # MANN core model
    mann_core = MANN(
        input_size=config["embedding_dim"] + config["num_classes"],
        hidden_size=config["hidden_size"],
        memory_size=config["memory_size"],
        word_size=config["word_size"],
        output_size=config["num_classes"],
        num_read_heads=config["num_read_heads"],
        device=device
    )

    # MANN Wrapper - MANN + EEG Encoder
    model = MANNWrapper(
        mann=mann_core,
        eeg_encoder=eeg_encoder,
        num_classes=config["num_classes"],
        device=device
    )

    if mode == 'train':
        # Create meta-learning dataset for training
        # num_tasks specifies how many tasks (batches) are sampled per epoch
        train_meta_dataset = EEGMetaDataset(
            df=train_df,
            data_dir=config['data_dir'],
            num_tasks=config['num_tasks_train'],
            k_shot=config['k_shot'],
            q_query=config['q_query']
        )

         # Train the model using the MANN algorithm over specified epochs
        mann_train(
            model,
            train_meta_dataset,
            device,
            epochs=config["epochs"],
            lr=config["learning_rate"]
        )

        if config.get("save_model"):
            torch.save(model.state_dict(), config["save_model"])
            print(f"Model saved to {config['save_model']}")

    elif mode == 'test':
        if config.get("load_model") is None:
            raise ValueError("Please provide 'load_model' path in config for test mode.")

        model.load_state_dict(torch.load(config["load_model"]))
        print(f"Loaded model from {config['load_model']}")

        # Create meta-learning dataset for testing
        test_meta_dataset = EEGMetaDataset(
            df=test_df,
            data_dir=config['data_dir'],
            num_tasks=config['num_tasks_test'],
            k_shot=config['k_shot'],
            q_query=config['q_query']
        )

        # Test the model on unseen data
        mann_test(
            model,
            test_meta_dataset,
            device
        )
    
    else:
        raise ValueError("Mode must be either 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MANN with EEG Data using YAML Config")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)