import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Computes attention weights from similarity scores for Matching Networks.

    In Matching Networks, attention is used to weigh support examples based on 
    their similarity to each query. This module applies a softmax to normalize
    the similarity scores into attention weights.
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, similarities):
        """
        Forward pass to compute attention weights.

        Args:
            similarities (Tensor): 
                A 2D tensor of shape [Q, K], where:
                - Q is the number of query examples
                - K is the number of support examples
                Each element (i, j) in the tensor represents the similarity between
                the i-th query and the j-th support example.

        Returns:
            attention_weights (Tensor): 
                A 2D tensor of shape [K, Q], where each row is a softmax-normalized 
                distribution over the support examples. These serve as attention weights 
                to compute the final label prediction for each query.
        """
        # Apply softmax over the support dimension (K) for each query
        # This ensures the attention weights for each query sum to 1
        attention = F.softmax(similarities, dim=1)

        return attention