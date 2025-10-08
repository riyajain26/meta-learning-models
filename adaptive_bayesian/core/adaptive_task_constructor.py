import torch 
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTaskConstructor(nn.Module):
    """
    Adaptive Task Construction Module
    Generates meta-parameters and selectively constructs support sets
    based on query sample characteristics
    """

    def __init__(self, feature_dim=128, meta_param_rank=32, temperature_init=1.0, temperature_min=0.1):
        """
        Args:
            feature_dim: Dimension of time-frequency representation (128)
            meta_param_rank: Rank for low-rank factorization of meta-parameters
            temperature_init: Initial Gumbel temperature
            temperature_min: Minimum Gumbel temperature
        """
        super(AdaptiveTaskConstructor, self).__init__()
        
        self.feature_dim = feature_dim
        self.meta_param_rank = meta_param_rank
        self.temperature = temperature_init
        self.temperature_min = temperature_min
        
        # Low-rank factorization matrices P and Q
        # ε_k = P · D(z̃_k) · Q
        # Input: feature_dim → Output: meta_param_rank × meta_param_rank
        self.P = nn.Parameter(torch.randn(meta_param_rank, feature_dim) * 0.01)
        self.Q = nn.Parameter(torch.randn(feature_dim, meta_param_rank) * 0.01)
        
        # Chain predictor for computing selection probabilities
        # Input: concatenated [z_i, z̃_k] → Output: probability
        self.selection_network = nn.Sequential(
            nn.Linear(meta_param_rank * meta_param_rank, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim * 2)
        )

        # Learnable bias for the selection function
        self.selection_bias = nn.Parameter(torch.zeros(1))

    def generate_meta_parameters(self, query_representation):
        """
        Generate meta-parameters ε_k from query representation
        ε_k = P · D(z̃_k) · Q
        
        Args:
            query_representation: (feature_dim,)
        Returns:
            meta_params: (batch_size, meta_param_rank, meta_param_rank) or 
                        (meta_param_rank, meta_param_rank)
        """
        
        single = False
        if query_representation.dim() == 1:
            query_representation = query_representation.unsqueeze(0)
            single = True
        
        # batch_size=1 always
        batch_size = query_representation.size(0)
        
        # Create diagonal matrix D(z̃_k)
        # For each sample in batch: diag(z̃_k)
        D = torch.diag_embed(query_representation)  # (batch_size, feature_dim, feature_dim)
        
        # Compute ε_k = P · D(z̃_k) · Q
        # P: (meta_param_rank, feature_dim)
        # D: (batch_size, feature_dim, feature_dim)
        # Q: (feature_dim, meta_param_rank)
        
        # Expand P for batch processing
        P_expanded = self.P.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, meta_param_rank, feature_dim)
        
        # P · D
        PD = torch.bmm(P_expanded, D)  # (batch_size, meta_param_rank, feature_dim)
        
        # (P · D) · Q
        Q_expanded = self.Q.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, feature_dim, meta_param_rank)
        meta_params = torch.bmm(PD, Q_expanded)  # (batch_size, meta_param_rank, meta_param_rank)
        
        if single:
            meta_params = meta_params.squeeze(0)
        
        return meta_params
    
    def compute_selection_probabilities(self, support_pool_representations, 
                                       query_representation, meta_params):
        """
        Compute selection probabilities ρ_i for each sample in support pool
        ρ_i = σ(ε_k · [z_i, z̃_k])
        
        Args:
            support_pool_representations: (N, feature_dim) - N samples in pool
            query_representation: (feature_dim,) - single query
            meta_params: (meta_param_rank, meta_param_rank) - meta-parameters
        Returns:
            probabilities: (N,) - selection probability for each sample
        """
        N = support_pool_representations.size(0)
        
        # Convert meta-parameters to selection weights
        # Flatten ε_k: (meta_param_rank, meta_param_rank) → (meta_param_rank^2,)
        meta_params_flat = meta_params.flatten()

        # Generate selection weights: (meta_param_rank^2,) → (feature_dim*2,)
        # This is the "chain predictor" that uses meta-parameters
        selection_weights = self.selection_network(meta_params_flat)  # (feature_dim*2,)

        # Expand query representation to match support pool
        query_expanded = query_representation.unsqueeze(0).expand(N, -1)  # (N, feature_dim)
        
        # Concatenate each support sample with query
        combined = torch.cat([support_pool_representations, query_expanded], dim=1)  # (N, feature_dim*2)
        
        # Compute scores using the meta-parameter-derived weights
        # combined: (N, 256), selection_weights: (256,)
        scores = torch.matmul(combined, selection_weights) + self.selection_bias  # (N,)
        
        # Step 4: Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(scores)
        
        return probabilities
    
    def gumbel_reparameterization(self, probabilities, temperature=None):
        """
        Apply Gumbel reparameterization trick for differentiable sampling
        V_i = σ((log(ρ_i/(1-ρ_i)) + (g¹_i - g²_i))/temperature)
        
        Args:
            probabilities: (N,) - selection probabilities
            temperature: scalar - Gumbel temperature (uses self.temperature if None)
        Returns:
            selections: (N,) - soft selection values in [0, 1]
        """
        if temperature is None:
            temperature = self.temperature
        
        # Clamp probabilities to avoid log(0) or division by zero
        probabilities = torch.clamp(probabilities, min=1e-7, max=1-1e-7)
        
        # Compute log odds: log(ρ_i / (1 - ρ_i))
        log_odds = torch.log(probabilities / (1 - probabilities))
        
        # Sample Gumbel noise
        gumbel_1 = -torch.log(-torch.log(torch.rand_like(probabilities) + 1e-10) + 1e-10)
        gumbel_2 = -torch.log(-torch.log(torch.rand_like(probabilities) + 1e-10) + 1e-10)
        gumbel_diff = gumbel_1 - gumbel_2
        
        # Apply Gumbel reparameterization
        selections = torch.sigmoid((log_odds + gumbel_diff) / temperature + 1e-10)
        
        return selections
    

    def select_support_set(self, support_pool_data, support_pool_labels,
                          support_pool_representations, query_representation,
                          meta_params, hard_threshold=0.5, training=True):
        """
        Select support set from support pool based on query
        
        Args:
            support_pool_data: (N, n_channels, time_length) - raw EEG data
            support_pool_labels: (N,) - labels
            support_pool_representations: (N, feature_dim) - representations
            query_representation: (feature_dim,) - query representation
            meta_params: (meta_param_rank, meta_param_rank) - meta-parameters
            hard_threshold: threshold for binary selection (default 0.5)
            training: if True, use Gumbel soft selection; if False, use hard selection
        Returns:
            selected_data: (K, n_channels, time_length) - selected samples
            selected_labels: (K,) - selected labels
            selected_representations: (K, feature_dim) - selected representations
            selection_mask: (N,) - selection values (soft or hard)
        """
        # Compute selection probabilities
        probabilities = self.compute_selection_probabilities(
            support_pool_representations, query_representation, meta_params
        )
        
        if training:
            # Use Gumbel soft selection during training
            selection_mask = self.gumbel_reparameterization(probabilities)
            # We'll return all data but with weights for the downstream network
            selected_data = support_pool_data
            selected_labels = support_pool_labels
            selected_representations = support_pool_representations
        else:
            # Use hard threshold during inference
            selection_mask = (probabilities > hard_threshold).float()
            # Hard selection for inference
            selected_indices = torch.where(selection_mask > hard_threshold)[0]
            
            if len(selected_indices) == 0:
                # If nothing selected, take top-k
                k = min(10, len(probabilities))
                selected_indices = torch.topk(probabilities, k).indices
            
            selected_data = support_pool_data[selected_indices]
            selected_labels = support_pool_labels[selected_indices]
            selected_representations = support_pool_representations[selected_indices]
        
        return selected_data, selected_labels, selected_representations, selection_mask
    
    def update_temperature(self, epoch, total_epochs):
        """
        Gradually decrease temperature during training
        
        Args:
            epoch: current epoch
            total_epochs: total number of epochs
        """
        # Linear annealing
        progress = epoch / total_epochs
        self.temperature = max(
            self.temperature_min,
            1.0 - progress * (1.0 - self.temperature_min)
        )
    
    def forward(self, support_pool_data, support_pool_labels,
               support_pool_representations, query_representation,
               training=True):
        """
        Complete forward pass: generate meta-params and select support set
        
        Args:
            support_pool_data: (N, n_channels, time_length)
            support_pool_labels: (N,)
            support_pool_representations: (N, feature_dim)
            query_representation: (feature_dim,)
            training: bool
        Returns:
            selected_data: selected support samples
            selected_labels: selected labels
            selected_representations: selected representations
            selection_mask: selection weights/mask
            meta_params: generated meta-parameters
        """
        # Generate meta-parameters
        meta_params = self.generate_meta_parameters(query_representation)
        
        # Select support set
        selected_data, selected_labels, selected_representations, selection_mask = \
            self.select_support_set(
                support_pool_data, support_pool_labels,
                support_pool_representations, query_representation,
                meta_params, training=training
            )
        
        return (selected_data, selected_labels, selected_representations, 
                selection_mask, meta_params)
    
