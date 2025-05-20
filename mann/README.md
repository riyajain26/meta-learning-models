# MANN: Memory-Augmented Neural Network

This folder contains an implementation of the MANN algorithm using PyTorch, tailored for few-shot EEG classification tasks by leveraging an external memory module.

This implementation is part of the `meta-learning-models` project.


## Table of Contents
- [Overview](#overview)
- [Algorithm](#algorithm)
- [Directory Structure](#directory-structure)
- [Implementation Detials](#implementation-details)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dataset](#dataset)
- [References](#references)


## Overview

Memory-Augmented Neural Networks (MANN) were introduced by Santoro et al. as a framework for meta-learning that integrates neural networks with external memory modules. This allows the network to learn to store and retrieve representations across tasks with minimal supervision.

This repo supports:
- Few-shot EEG classification using memory-augmented architecture
- Plug-and-play EEG dataset integration
- Clear and modular implementation

## Algorithm

The core idea of MANN is to use an LSTM-based controller with an external memory to perform few-shot learning. The network:
- Stores information from support examples in memory.
- Uses attention-based memory access to retrieve useful information during query classification.
- Learns a learning algorithm through backpropagation across episodes.

This implementation is based on the architecture described in the [original paper](https://arxiv.org/pdf/1605.06065).

## Directory Structure

```bash
mann/
├── core/ # MANN architecture and core logic
├── config/ # YAML config files for training/testing
├── utils/ # Helper functions
├── main.py # Entry point for training/testing
├── README.md # This file
```

## Implementation Details

This implementation of MANN has been adapted for EEG signal classification. The main components are organized as follows:

### core/eeg_encoder.py

- Contains the CNN-based feature extractor for EEG signals.
- Converts raw EEG input (shape: channels × time_steps) into a compact embedding vector.
- This encoder is used before feeding data into the actualy MANN model (controller + memory module).
- Customizable architecture with convolutional layers, batch normalization, and ReLU activations.
- This is an add-on over basic MANN model to process the EEG signals. 

### core/eeg_meta_dataset.py

- Defines the data loading pipeline for few-shot EEG classification.
- Dynamically samples tasks (episodes) from the dataset with N classes and K support/Q query examples per class.
- Provides support and query sets for each task to train and evaluate MANN.
- Supports EEG .npy files and loads labels and participant/session metadata from metadata.csv.

### core/lstm_controller.py

- Implements an LSTM-based controller which acts as the central processing unit of the MANN.
- Receives the EEG embeddings as input (output from eeg_encoder) and interacts with the external memory.
- Generates output vectors that are used to compute attention weights for memory read/write operations.

### core/memory_module.py

- Contains the external memory mechanism with read and write operations.
- Supports content-based addressing (similarity-based memory access).
- Facilitates few-shot learning by enabling rapid storage and retrieval of support set information.
- Implements reading mechanism based on Equations 2-4 as given in the [original paper](https://arxiv.org/pdf/1605.06065).
- Implements writing mechanism based on Equations 5-8 as given in the [original paper](https://arxiv.org/pdf/1605.06065).

### core/mann_model.py

- Assembles the complete MANN architecture.
- Combines the encoder, LSTM controller, and memory module into a single forward pipeline.

### core/train.py

- Handles the training loop for episodic learning.
- For each meta-training episode:
    - Loads a new task.
    - Feeds support and query samples through encoder and writes to memory.
    - Computes loss for query samples.
- Optimizes the full MANN model and logs training loss.

### core/test.py

- Similar to the training pipeline but without gradient updates.
- Loads unseen tasks from the meta-test set.
- Evaluates generalization performance on query samples after memory initialization from support set.
- Computes per-task loss and averages results across multiple test episodes.

## Installation

```bash
git clone https://github.com/riyajain26/meta-learning-models.git
cd meta-learning-models
pip install -r requirements.txt
```

## Configuration

Modify/Create the YAML files in the config/ directory to adjust parameters/hyperparameters like:
- Dataset and Training configuration
- EEG Signal/Model Parameters
- Meta-dataset Parameters
- Training configuration
- Model save/load information

## Usage

### Training

```bash
python3 -m mann.main --mode train --config mann/config/config_template.yaml
```

### Testing

```bash
python3 -m mann.main --mode test --config mann/config/config_template.yaml
```

## Dataset

It supports any type of EEG dataset with consistent shape (channels, time_steps) across datapoints. The datasets are stored in `datasets/` directory. 

Make sure to add `metadata.csv` that contains information about eeg_samples (.npy), lables, and participants. A few dummy datasets have been added for the reference.

## References

Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/pdf/1605.06065)