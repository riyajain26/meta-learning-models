import torch
import torch.nn as nn
import math

from snail.core.blocks import *     # Contains AttentionBlock, TCBlock, etc.

class SNAIL(nn.Module):
    def __init__(self, num_classes, in_channels, time_length, kernel_size):
        """
        SNAIL model for meta-learning with EEG signals.
        Args:
            num_classes (int): Number of classification outputs
            in_channels (int): Dimensionality of input (embedding + label vector)
            time_length (int): Total number of time steps (K + Q)
            kernel_size (int): Kernel size for causal convolutions
        """

        super(SNAIL,self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels      # Initial input = embedding dim + num_classes
        self.time_length = time_length 
        self.kernel_size = kernel_size

        # Number of DenseBlocks in each TCBlock (log2 of sequence length)
        self.num_filters = int(math.ceil(math.log(time_length,2)))

        # --- Attention Block 1 ---
        self.attention1 = AttentionBlock(in_channels=self.in_channels, key_size=64, val_size=32)
        self.in_channels += 32      # Output of attention is concatenated

        # --- Temporal Convolution Block 2 ---
        self.tc2 = TCBlock(in_channels=self.in_channels, time_length=self.time_length, filters=128, kernel_size=2)
        self.in_channels += self.num_filters * 128      # TCBlock adds filters * num_layers

        # --- Attention Block 2 ---
        self.attention2 = AttentionBlock(in_channels=self.in_channels, key_size=256, val_size=128)
        self.in_channels += 128

        # --- Temporal Convolution Block 3 ---
        self.tc3 = TCBlock(in_channels=self.in_channels, time_length=self.time_length, filters=128, kernel_size=2)
        self.in_channels += self.num_filters * 128

        # --- Attention Block 3 ---
        self.attention3 = AttentionBlock(in_channels=self.in_channels, key_size=512, val_size=256)
        self.in_channels += 256

        # --- Final Fully Connected Layer ---
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x):
        """
        Forward pass of the SNAIL model.
        Args:
            x (Tensor): Shape (B, T, C) = (1, T=K+Q, embedding_dim + num_classes)
        Returns:
            logits: (B, T, num_classes)
        """
        x = self.attention1(x)
        x = self.tc2(x)
        x = self.attention2(x)
        x = self.tc3(x)
        x = self.attention3(x)

        logits = self.fc(x)
        return logits


# --- Wrapper for Encoder + SNAIL ---
class SNAILWrapper(nn.Module):
    def __init__(self, encoder, snail):
        """
        Combines an encoder (e.g., EEGEmbeddingNet) with the SNAIL model.
        Args:
            encoder (nn.Module): Embedding network for EEG input
            snail (SNAIL): The SNAIL meta-learner model
        """
        super(SNAILWrapper, self).__init__()
        self.encoder = encoder
        self.snail = snail

    def forward(self, inputs, labels):
        """
        Args:
            inputs: EEG signals of shape (N=T, C, L)
            labels: One-hot encoded labels of shape (N=T, num_classes)
        Returns:
            logits: Model predictions (N=T, num_classes)
        """
        # Encode EEG data → shape: (N, embedding_dim)
        embedded_inputs = self.encoder(inputs)

        # Concatenate embeddings with one-hot labels → (N, embedding_dim + num_classes)
        snail_input = torch.cat([embedded_inputs, labels], dim=-1) 

        # Add batch dimension for SNAIL input → (1, T, C)
        snail_input = snail_input.unsqueeze(0)
        
        # Forward through SNAIL → output shape: (1, T, num_classes)
        logits = self.snail(snail_input)

        # Remove batch dimension → (T, num_classes)
        logits = logits.squeeze(0)

        return logits
