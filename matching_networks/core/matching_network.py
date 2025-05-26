import torch
import torch.nn as nn
import torch.nn.functional as F

from matching_networks.core.attention import Attention
from matching_networks.core.similarity import CosineSimilarity


class MatchingNetwork(nn.Module):
    """
    Matching Network for Meta-Learning (Few-Shot Learning).

    This model performs classification of query examples by computing similarity 
    to a labeled support set in an embedding space, and using an attention mechanism 
    to weight the support labels accordingly.

    Particularly suited for EEG signal classification where data is limited per class.
    """
    def __init__(self, encoder, embedding_dim=128, num_classes=4):
        """
        Initializes the Matching Network model.

        Args:
            encoder: Neural network (e.g., CNN for EEG) that maps inputs to embeddings.
            embedding_dim: Dimensionality of the learned embeddings.
            num_classes: Number of classes in the classification task.
        """

        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.similarity_fn = CosineSimilarity()             # Module to compute similarity scores
        self.attention = Attention()                        # Module to compute attention weights

    def forward(self, support_x, support_y, query_x):
        """
        Forward pass of the Matching Network.

        Args:
            support_x (Tensor): Support set inputs of shape [K, C, T]
                                K = # of support samples = N_way*K_shot
                                C = # of EEG channels
                                T = time steps (e.g., 1875)
            support_y (Tensor): Labels for support set of shape [K]
            query_x   (Tensor): Query set inputs of shape [Q, C, T]
                                Q = # of query samples = N_way*Q_query

        Returns:
            preds (Tensor): Predicted softmax scores of shape [Q, num_classes]
        """

        # Step 1: Encode support and query samples to embedding space
        # z_support: [K, D], z_query: [Q, D]
        z_support = self.encoder(support_x)
        z_query = self.encoder(query_x)

        # Step 2: Compute similarity scores between each query and all support examples
        # similarities: [Q, K]
        similarities = self.similarity_fn(z_query, z_support)

        # Step 3: Convert similarity scores to attention weights (softmax over support examples)
        # attention_weights: [Q, K]
        attention_weights = self.attention(similarities)

        # Step 4: Convert support labels to one-hot encoded vectors
        # one_hot_labels: [K, num_classes]
        one_hot_labels = F.one_hot(support_y, num_classes=self.num_classes).float()

        # Step 5: Predict query labels as weighted sum of support labels
        # preds: [Q, num_classes]
        preds = torch.matmul(attention_weights, one_hot_labels)

        return preds