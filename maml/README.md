# MAML: Model-Agnostic Meta-Learning

This folder contains an implementation of the MAML algorithm using PyTorch, which enables models to quickly adapt to new tasks with only a few gradient steps. 

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

Model-Agnostic Meta-Learning (MAML) is a meta-learning algorithm introduced by Chelsea Finn et al., which learns an initialization of model parameters that can be fine-tuned quickly on new tasks.

This repo supports:
- Few-shot classification on different EEG datasets
- Easy integration of new datasets
- Modular and readable codebase

## Algorithm

MAML optimizes for a model initialization that can quickly adapt to new tasks using a small number of gradient steps. It consists of two loops:
- **Inner loop**: Adapts model parameters to a specific task.
- **Outer loop**: Updates the initialization based on performance across tasks.

The implementation is based on Algorithm 2 described in the [original paper](https://arxiv.org/pdf/1703.03400).

## Directory Structure

```bash
maml/
├── core/ # Model architecture and MAML core
├── config/ # YAML config files for training/testing
├── utils/ # Helper functions
├── main.py # Entry point for training/testing
├── README.md # This file
```

## Implementation Details

This implementation of MAML has been adapted for EEG signal classification. The main components are organized as follows:

### core/eeg_cnn.py

- Defines the model whose initial parameter are learnt that can be fine-tuned quickly on new unseen tasks.
- Defines the CNN-based classifier used for EEG signals. 
- A different model can be built and used for EEG classification (if needed). 
- The model architecture is lightweight and designed to capture spatiotemporal features from EEG input data.
- It supports customization for the number of EEG channels and time steps.
- Implements a modular design to easily integrate with the MAML training loop.

### core/eeg_meta_dataset.py

- Defines the data loading pipeline for few-shot EEG classification.
- Dynamically samples tasks (episodes) from the dataset with N classes and K support/Q query examples per class.
- Provides support and query sets for each task to train and evaluate MAML.
- Supports EEG .npy files and loads labels and participant/session metadata from metadata.csv.

### core/train.py

- Contains the core MAML training algorithm. 
- Samples batches of tasks from the meta-dataset and performs training loops.
- Implements:
    - Inner loop: Fast gradient-based adaptation on support sets.
    - Outer loop: Meta-optimization across tasks using query set losses.
- Supports full second-order MAML.
- Tracks task-level loss and gradients for meta-update computation.

### core/test.py

- Used for evaluating the trained model on unseen EEG tasks.
- Runs few-shot adaptation using support sets.
- Evaluates performance on corresponding query sets.
- Reports per-task and averaged classification loss.

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
python3 -m maml.main --mode train --config maml/config/config_template.yaml
```

### Testing

```bash
python3 -m maml.main --mode test --config maml/config/config_template.yaml
```

## Dataset

It supports any type of EEG dataset with consistent shape (channels, time_steps) across datapoints. The datasets are stored in `datasets/` directory. 

Make sure to add `metadata.csv` that contains information about eeg_samples (.npy), lables, and participants. A few dummy datasets have been added for the reference.

## References

Finn, C., Abbeel, P., & Levine, S. (2017). [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400)