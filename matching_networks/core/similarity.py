import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
    """
    Computes similarity between query and support embeddings.

    Currently supports cosine similarity, but can be extended
    to other metrics like Euclidean or learnable similarity.
    """
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, z_query, z_support):
        """
        Args:
            z_query: Tensor of shape [N_q, D] - embeddings of query examples
            z_support: Tensor of shape [N_s, D] - embeddings of support examples

        Returns:
            similarity_matrix: Tensor of shape [N_q, N_s] - pairwise similarity scores
        """
        # Expand dimensions to compute pairwise cosine similarity between all query-support pairs
        similarity = F.cosine_similarity(
            z_query.unsqueeze(1),    # [N_q, 1, D]
            z_support.unsqueeze(0),  # [1, N_s, D]
            dim=2                    # Compute similarity along embedding dimension
        )

        return similarity