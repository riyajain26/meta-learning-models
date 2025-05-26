# Matching Networks for Few-shot EEG Classification

This folder contains an implementation of the Matching Networks algorithm using PyTorch, adapted for few-shot EEG signal classification tasks. Matching Networks use metric learning to classify new samples based on similarity to a small support set.

This implementation is part of the `meta-learning-models` project.


## Table of Contents
- [Overview](#overview)
- [Algorithm](#algorithm)
- [Directory Structure](#directory-structure)
- [Implementation Detials](#implementation-details)
- [Configuration](#configuration)
- [Usage](#usage)
- [References](#references)


## Overview

Matching Networks were introduced by Vinyals et al. to perform classification by comparing embeddings of query samples with those of support examples using a cosine similarity metric.

This repo supports:
- Few-shot EEG classification using Matching Networks
- Modularity to plug in various EEG datasets
- Clear and extensible code structure

## Algorithm

The Matching Network classifies query examples by comparing their embeddings to those of a small support set and computing a weighted sum of the support labels using softmax over cosine similarities.

Key ideas:
- End-to-end differentiable architecture trained using episodic learning
- Embedding function shared by both support and query examples
- Attention kernel based on cosine similarity
- No need for explicit memory module

This implementation is based on the [original paper](https://arxiv.org/pdf/1606.04080).

## Directory Structure

```bash
mann/
├── core/ # Matching Network core logic
├── config/ # YAML config files for training/testing
├── utils/ # Helper functions
├── main.py # Entry point for training/testing
├── README.md # This file
```

## Implementation Details

This implementation of Matching Network has been adapted for EEG signal classification. The main components are organized as follows:

### core/eeg_encoder.py

- Contains the CNN-based feature extractor for EEG signals.
- Converts raw EEG input (shape: channels × time_steps) into a compact embedding vector.
- Customizable architecture with convolutional layers, batch normalization, and ReLU activations.

### core/eeg_meta_dataset.py

- Defines the data loading pipeline for few-shot EEG classification.
- Dynamically samples tasks (episodes) from the dataset with N classes and K support/Q query examples per class.
- Provides support and query sets for each task to train and evaluate Matching Network.
- Supports EEG .npy files and loads labels and participant/session metadata from metadata.csv.

### core/attention.py

- Implements the attention mechanism used for computing similarities between support and query embeddings.
- Generates attention weights (softmax-normalized cosine similarities) that are used to compute the final prediction as a weighted sum of support labels.


### core/similarity.py

- Contains functions to compute pairwise cosine similarity between embedded EEG support and query examples.
- Used as the similarity measure in attention calculation as described in the Matching Networks paper.

### core/matching_network.py

- Assembles the full Matching Network architecture for few-shot classification.
- Processes support and query samples through a shared encoder.
- Computes cosine similarity between embeddings and applies attention mechanism to produce query predictions.

### core/train.py

- Episodic training loop for Matching Networks.
- For each episode:
    - Samples a new task
    - Embeds support and query sets
    - Computes prediction via cosine attention
    - Backpropagates query loss and updates model

### core/test.py

- Similar to the training pipeline but without gradient updates.
- Loads unseen tasks from the meta-test set.
- Evaluation pipeline for few-shot tasks using the trained Matching Network.

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
python3 -m matching_networks.main --mode train --config matching_networks/config/config_template.yaml
```

### Testing

```bash
python3 -m matching_networks.main --mode test --config matching_networks/config/config_template.yaml
```

## References

Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080)