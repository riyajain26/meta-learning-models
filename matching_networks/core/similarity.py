import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
    """
    Computes pairwise similarity between query and support embeddings.

    This implementation uses cosine similarity, which is often preferred 
    in few-shot learning tasks (like Matching Networks) due to its 
    scale-invariant nature and its ability to measure the angular closeness 
    between feature vectors.
    """
    def __init__(self):
        """
        Initializes the CosineSimilarity module.
        Currently no parameters are needed, but structure allows easy extension 
        for alternative similarity metrics (e.g., Euclidean, dot product, learnable similarity).
        """
        super(CosineSimilarity, self).__init__()

    def forward(self, z_query, z_support):
        """
        Forward pass to compute cosine similarity between all query and support pairs.

        Args:
            z_query (Tensor): Embeddings of query examples.
                              Shape: [Q, D] where Q is number of queries, D is embedding dim.
            z_support (Tensor): Embeddings of support examples.
                                Shape: [K, D] where K is number of supports.

        Returns:
            similarity_matrix (Tensor): Pairwise similarity scores.
                                        Shape: [Q, K] where entry (i, j) is similarity between
                                        i-th query and j-th support.
        """
        # Step 1: Expand z_query and z_support to enable broadcasting
        # z_query: [Q, 1, D], z_support: [1, K, D]
        # This sets up pairwise combinations of query-support embeddings

        # Step 2: Compute cosine similarity along the embedding dimension (D)
        # Resulting tensor shape: [Q, K]
        # Expand dimensions to compute pairwise cosine similarity between all query-support pairs
        similarity = F.cosine_similarity(
            z_query.unsqueeze(1),    # Add dimension for broadcasting: [Q, 1, D]
            z_support.unsqueeze(0),  # Add dimension for broadcasting: [1, K, D]
            dim=2                    # Compute similarity along embedding dimension
        )

        return similarity