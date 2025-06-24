import torch
import torch.nn as nn
import torch.nn.functional as F



class MAMLWrapper(nn.Module):
    def __init__(self, eeg_encoder, embedding_dim, num_classes, device):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.device = device

    def forward(self, eeg_seq):
        features = self.eeg_encoder(eeg_seq)
        out = self.classifier(features)
        return out
