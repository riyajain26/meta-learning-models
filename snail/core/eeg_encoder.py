import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEmbeddingNet(nn.Module):
    """
    EEG Encoder for learning compact feature embeddings from EEG signals.

    It is designed to encode EEG signals into a fixed-size embedding. 
    It transforms an input EEG signal of shape (batch_size, in_channels, input_time) 
    into a feature vector of dimension `embedding_dim`.

    This encoder can be used as a feature extractor for tasks such as few-shot learning
    with SNAIL or other downstream classification tasks.
    """
    def __init__(self, embedding_dim=128, in_channels=22, input_time=1875):
        """
        Initializes the EEGEmbeddingNet.

        Args:
            embedding_dim (int): Dimension of the final embedding output.
            in_channels (int): Number of EEG channels in the input (default: 22).
            input_time (int): Length of the input time series per channel (default: 1875).
        """
        super(EEGEmbeddingNet, self).__init__()
        
        # Layer 1: Conv1D + BatchNorm + MaxPooling
        # Input:  (B, 22, 1875)
        # Output: (B, 32, 937)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2: Conv1D + BatchNorm + MaxPooling
        # Input:  (B, 32, 937)
        # Output: (B, 64, 468)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Layer 3: Conv1D + BatchNorm + AdaptiveAvgPool
        # Input:  (B, 64, 468)
        # Output: (B, 128, 1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)    ## Shape: (B, 128, 468)
        self.bn3 = nn.BatchNorm1d(128) ## Shape: (B, 128, 468)
        
        # We'll use this to compress the output size  
        # Calculate the output size after convolutions
        # This avoids hard-coding the size which might change with different inputs
        self.adaptive_pool3 = nn.AdaptiveAvgPool1d(1)  ## Shape: (B, 128, 1)

        # Fully connected layer to produce feature embeddings
        # Input:  (B, 128)
        # Output: (B, embedding_dim)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the EEGEmbeddingNet.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, input_time)

        Returns:
            Tensor: Feature embedding of shape (B, embedding_dim)
        """

        # Pass through first convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool3(x)      # Output shape: (B, 128, 1)

        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)       # Flatten to shape (B, 128)
        x = self.fc(x)          
        
        return x                        # Returns shape: (B, embedding_dim)