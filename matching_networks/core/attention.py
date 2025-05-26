import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Computes attention weights from similarity scores.

    Applies a softmax over the support examples for each query.
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, similarities):
        """
        Args:
            similarities: Tensor of shape [N_q, N_s] - similarity scores between queries and supports

        Returns:
            attention_weights: Tensor of shape [N_q, N_s] - normalized attention weights
        """
        
        attention = F.softmax(similarities, dim=1)
        return attention