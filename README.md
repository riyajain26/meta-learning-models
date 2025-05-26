# Meta-Learning Models for Few-Shot EEG Classification

This repository contains modular and extensible implementations of state-of-the-art meta-learning algorithms for few-shot EEG classification using PyTorch.

Each algorithm is adapted to handle EEG signals and is organized into its own subdirectory with separate configurations, models, and training pipelines.


## Table of Contents
- [Implemented Algorithms](#implemented-algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)


## Implemented Algorithms

| Model | Description | Directory |
|-------|-------------|-----------|
| **MAML** | Model-Agnostic Meta-Learning. Learns a good initialization that can be fine-tuned with a few gradient steps. | [`maml/`](./maml) |
| **MANN** | Memory-Augmented Neural Network. Uses an external memory module with an LSTM controller for one-shot learning. | [`mann/`](./mann) |
| **Matching Network** | Matching Network. Metric-based model using cosine similarity and attention over support embeddings to classify queries. | [`matching_network/`](./matching_network) |

Each model has its own README file with in-depth documentation on usage, architecture, and references.

## Project Structure

```bash
meta-learning-models/
├── maml/ # MAML implementation
├── mann/ # MANN implementation
├── matching_network/ # Matching Network implementation
├── datasets/ # EEG data and metadata
├── checkpoints/ # Save checkpoints
├── requirements.txt # Python dependencies
└── README.md # This file
```

- Place EEG datasets under the `datasets/` directory. Dummy datasets are included as a reference. Each dataset should include:
    - .npy files for EEG samples
    - metadata.csv with sample names, labels, and participants
- Each model supports any type of EEG dataset with consistent shape (channels, time_steps) across datapoints.
- Model checkpoints after training are saved in `checkpoints/` directory. Saved models can be loaded during testing phase.

## Installation

```bash
git clone https://github.com/riyajain26/meta-learning-models.git
cd meta-learning-models
pip install -r requirements.txt
```