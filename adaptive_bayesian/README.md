# ABML: Adaptive Bayesian Meta-Learning

This folder contains an implementation of the Adaptive Bayesian Meta-Learning (ABML) algorithm using PyTorch, which enables rapid adaptation to heterogeneous EEG signals through Bayesian posterior estimation and adaptive task construction. 

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

Adaptive Bayesian Meta-Learning (ABML) is a meta-learning framework designed to address the challenges of data heterogeneity in EEG signal classification. Unlike traditional meta-learning approaches, ABML introduces:

- Bayesian Meta-Learning: Treats task-specific parameters as latent variables inferred via amortized variational inference
- Adaptive Task Construction: Dynamically selects relevant support samples for each query using query-guided meta-parameters
- Time-Frequency Representation Learning: Dual VAE architecture that learns disentangled time and frequency representations

This repo supports:

- Few-shot classification on time-synchronous and time-asynchronous EEG datasets
- Adaptive support set selection based on query characteristics
- Fast inference through Bayesian posterior estimation
- Easy integration of new EEG datasets
- Modular and optimized codebase

## Algorithm

ABML consists of three main components working together to handle EEG data heterogeneity:
### Time-Frequency Enhanced Representation Learning
- Uses dual Variational Autoencoders (VAEs) to extract time and frequency domain features
- Applies information bottleneck constraints to maximize task-relevant information
- Produces disentangled representations: z = [z_T, z_F]

### Adaptive Task Construction

- Generates query-specific meta-parameters: ε_k = P · D(z̃_k) · Q
- Selects relevant support samples using Gumbel-softmax reparameterization
- Constructs task-specific support sets tailored to each query

### Amortized Variational Inference Network

- Generates Bayesian posterior for task-specific parameters: q(θ_k | D_k)
- Computes class prototypes via weighted average pooling
- Makes predictions: ỹ_k = θ_k · z̃_k

TThe implementation follows the methodology described in the [original paper](https://ieeexplore.ieee.org/document/10386001).

## Directory Structure

```bash
adaptive_bayesian/
├── core/ # Model architecture and ABML core
├── config/ # YAML config files for training/testing
├── utils/ # Helper functions
├── main.py # Entry point for training/testing
├── README.md # This file
```

## Implementation Details

This implementation of ABML has been adapted and optimized for EEG signal classification. The main components are organized as follows:

### core/tfrl_module.py

- Time-Frequency Representation Learning Module: Implements dual VAE architecture
- Extracts time-domain features (z_T) and frequency-domain features (z_F) using shared encoder
- Applies Fourier transform for frequency domain reconstruction constraints
- Implements mutual information maximization: I(y, z_T) + I(y, z_F) - I(z_T, z_F)
- Produces concatenated representation z = [z_T, z_F] with dimension 2 × latent_dim
- Supports both training mode (with losses) and inference mode (representation only)

### core/adaptive_task_constructor.py

- Adaptive Task Construction Module: Dynamically selects support samples per query
- Generates query-specific meta-parameters via low-rank factorization: ε_k = P · D(z̃_k) · Q
- Computes selection probabilities using meta-parameter-driven chain predictor
- Applies Gumbel-softmax reparameterization for differentiable sampling during training
- Uses temperature annealing (1.0 → 0.1) for gradually sharpening selections
- Supports both soft selection (training) and hard selection (inference)

### core/amortized_variational_inference_network.py

- Amortized Variational Inference Network: Generates Bayesian posteriors for task parameters
- Computes class prototypes from support set via weighted average pooling
- Generates posterior distribution parameters: q(θ_k | D_k)
- Samples task-specific parameters using reparameterization trick
- Makes predictions via linear projection: ỹ_k = θ_k · z̃_k
- Maintains fallback prototypes for handling missing classes

### core/abml.py

- Complete ABML Model: Integrates all three components into end-to-end architecture
- Implements full forward pass following Algorithm 1 (training) and Algorithm 2 (testing)
- Computes complete loss function: L_ABML = L(ω, φ) + α·L_TFRL
- Handles batch processing of multiple query samples
- Provides both training mode (with gradients) and inference mode (fast prediction)

### core/abml_trainer.py

- Training Loop: Implements meta-training and evaluation procedures
- Samples support pools and query batches for each training iteration
- Updates model parameters using Adam optimizer with gradient clipping
- Optimized evaluation: Pre-computes support representations once, reuses for all queries
- Processes test samples in batches (up to 128 at once) for efficiency
- Implements temperature annealing schedule for Gumbel-softmax
- Tracks training history and saves best models based on test accuracy

### core/eeg_meta_dataset.py

- EEG Dataset Handler: Manages EEG data organized by subjects
- Stores data as (N, in_channels, time_length) tensors with corresponding labels and subject IDs
- Provides methods to retrieve data for specific subjects or groups of subjects
- Supports leave-one-subject-out cross-validation paradigm
- Compatible with standard PyTorch DataLoader for batch processing

## Configuration

Modify/Create the YAML files in the config/ directory to adjust parameters/hyperparameters like:
- Dataset and Signal Parameters
- Model Architecture Parameters
- Training configuration
- Loss Weights
- Evaluation Parameters
- Task Constructor Parameters
- Model save/load information

## Usage

```bash
python3 -m adaptive_bayesian.main --config adaptive_bayesian/config/config_template.yaml
```

## References

X. Guo, J. Zhu, L. Zhang, B. Jin and X. Wei (2023). [Adaptive Bayesian Meta-Learning for EEG Signal Classification](https://ieeexplore.ieee.org/document/10386001) 
