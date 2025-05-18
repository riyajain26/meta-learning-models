import torch
import pandas as pd
import argparse
import yaml
from sklearn.model_selection import train_test_split


from maml.core.eeg_cnn import EEGCNN
from maml.core.eeg_meta_dataset import EEGMetaDataset
from maml.core.train import maml_train
from maml.core.test import maml_test
from maml.utils.utils import set_seed


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(metadata_path, test_size=0.2, random_state=42):
    # Load metadata CSV containing EEG recordings info (participant, labels, file paths, etc.)
    df = pd.read_csv(metadata_path)

    # Get unique participants and split into train/test groups (80% train, 20% test)
    participants = df['participant'].unique()
    train_participants, test_participants = train_test_split(participants, test_size=0.2, random_state=42)

    # Filter the DataFrame to create train and test DataFrames based on participant splits
    train_df = df[df['participant'].isin(train_participants)]
    test_df = df[df['participant'].isin(test_participants)]

    return train_df, test_df






def main(config):
    set_seed(42)
    # Select device: use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df = load_data(config['metadata_path'], test_size=config['test_split'])

    # Initialize the EEG classification model
    # Parameters: number of classes, number of EEG channels, length of EEG time series
    model = EEGCNN(
        num_classes=config['num_classes'], 
        in_channels=config['in_channels'], 
        input_time=config['input_time']
        ).to(device)


    if config['mode'] == 'train':
        # Create meta-learning dataset for training
        # num_tasks specifies how many tasks (batches) are sampled per epoch
        train_meta_dataset = EEGMetaDataset(
            df=train_df,
            data_dir=config['data_dir'],
            num_tasks=config['num_tasks'],
            k_shot=config['k_shot'],
            q_query=config['q_query']
        )

        # Train the model using the MAML algorithm over specified epochs
        maml_train(
            model, 
            train_meta_dataset,
            device, 
            epochs=config["epochs"],
            inner_steps=config["inner_steps"],
            inner_lr=config["inner_lr"],
            meta_lr=config["meta_lr"]
        )
        
        if config.get("save_model"):
            torch.save(model.state_dict(), config["save_model"])
            print(f"Model saved to {config['save_model']}")

    elif config["mode"] == "test":
        if config.get("load_model") is None:
            raise ValueError("Please provide 'load_model' path in config for test mode.")

        model.load_state_dict(torch.load(config["load_model"]))
        print(f"Loaded model from {config['load_model']}")

        # Create meta-learning dataset for testing
        test_meta_dataset = EEGMetaDataset(
            df=test_df,
            data_dir=config['data_dir'],
            num_tasks=config['num_tasks'],
            k_shot=config['k_shot'],
            q_query=config['q_query']
        )

        # Test the model on unseen data
        maml_test(
            model, 
            test_meta_dataset,
            device, 
            epochs=config["epochs"],
            inner_steps=config["inner_steps"],
            inner_lr=config["inner_lr"]
        )

    else:
        raise ValueError("Mode must be either 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAML with EEG Data using YAML Config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
