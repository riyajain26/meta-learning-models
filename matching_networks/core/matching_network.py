import torch
import torch.nn
import torch.nn.functional as F

from matching_networks.core.attention import Attention
from matching_networks.core.similarity import CosineSimilarity


class MatchingNetwork(nn.Module):
    """
    Matching Network for meta-learning.

    Predicts labels for query examples based on attention-weighted similarity to labeled support examples.
    """
    def __init__(self, encoder, embedding_dim=128, num_classes=4):
        """
        Args:
            encoder: Feature extractor (e.g., 1D CNN or RNN) that maps input to embedding space
            embedding_dim: Dimensionality of embedding space (D)
            num_classes: Number of possible class labels
        """
        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.similarity_fn = CosineSimilarity(metric='cosine')  # Module to compute similarity scores
        self.attention = Attention()                      # Module to compute attention weights

    def forward(self, support_x, support_y, query_x):
        """
        Forward pass of Matching Network.

        Args:
            support_x: Tensor of shape [N_s, C, T] - Support set inputs
            support_y: Tensor of shape [N_s]       - Support set labels
            query_x:   Tensor of shape [N_q, C, T] - Query set inputs

        Returns:
            preds: Tensor of shape [N_q, num_classes] - Soft class predictions for each query
        """
        # Encode support and query examples into embedding space
        z_support = self.encoder(support_x)  # [N_s, D]
        z_query = self.encoder(query_x)      # [N_q, D]

        # Compute pairwise similarity between queries and supports
        similarities = self.similarity_fn(z_query, z_support)  # [N_q, N_s]

        # Compute attention weights from similarity scores
        attention_weights = self.attention(similarities)        # [N_q, N_s]

        # Convert support labels to one-hot encoding for class prediction
        one_hot_labels = F.one_hot(support_y, num_classes=self.num_classes).float()  # [N_s, C]

        # Compute final predictions: weighted sum of support labels
        preds = torch.matmul(attention_weights, one_hot_labels)  # [N_q, C]

        return preds