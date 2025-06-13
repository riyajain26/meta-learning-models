import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    """
    Convolutional Neural Network for EEG Signal Classification.

    This model serves as the base learner for REPTILE.
    It is designed to quickly adapt to new EEG classification tasks with few examples.

    Args:
        num_classes (int): Number of output classes. Default is 2.
        in_channels (int): Number of EEG input channels (e.g., electrodes). Default is 30.
        input_time (int): Number of time steps in the input signal. Default is 384.
    """
    def __init__(self, num_classes=2, in_channels=30, input_time=384):
        super(EEGCNN, self).__init__()

        # In REPTILE (MAML), the model must be small and fast to adapt with limited data
        # Hence, this architecture is deliberately shallow and efficient

        # Layer 1: Conv1D + BatchNorm + MaxPooling
        # Input:  (B, 30, 384)
        # Output: (B, 32, 192)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2: Conv1D + BatchNorm + MaxPooling
        # Input:  (B, 32, 192)
        # Output: (B, 64, 96)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3: Conv1D + BatchNorm + AdaptiveAvgPool
        # Input:  (B, 64, 96)
        # Output: (B, 128, 1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(output_size=1)

        # Fully connected layer to produce class scores
        # Input:  (B, 128)
        # Output: (B, num_classes)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the EEGCNN model.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, input_time)

        Returns:
            Tensor: Class logits of shape (B, num_classes)
        """
        # Pass through first convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Output shape: (B, 128, 1)

        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to shape (B, 128)
        x = self.fc(x)             # Output shape: (B, num_classes)

        return x

