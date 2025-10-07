import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AmortizedVariationalInference(nn.Module):
    """
    Amortized Variational Inference Network
    Generates task-specific parameters θ_k from support set via Bayesian inference
    """

    def __init__(self, 
                 feature_dim=128,       # Concatenated z_T and z_F dimension
                 num_classes=4,           # Number of classes
                 hidden_dim=256,        # Hidden layer dimension
                 n_samples=1):          # Number of Monte Carlo samples
        """
        Args:
            feature_dim: Dimension of time-frequency representation (128)
            num_classes: Number of classification classes (4)
            hidden_dim: Hidden dimension for the inference network
            n_samples: Number of Monte Carlo samples for θ_k
        """
        super(AmortizedVariationalInference, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        
        # Network to generate posterior parameters from class prototypes
        # Input: class prototypes (num_classes, feature_dim)
        # Output: mean and log_variance for θ_k
        
        self.prototype_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Generate mean of posterior distribution
        self.posterior_mean = nn.Linear(hidden_dim, feature_dim)
        
        # Generate log variance of posterior distribution
        self.posterior_logvar = nn.Linear(hidden_dim, feature_dim)
        
        # For handling missing classes (fallback prototypes)
        self.register_buffer('fallback_prototypes', torch.zeros(num_classes, feature_dim))
        self.fallback_initialized = False
    
    def compute_class_prototypes(self, support_representations, support_labels, 
                                 selection_weights=None):
        """
        Compute class prototypes via average pooling
        
        Args:
            support_representations: (N, feature_dim) - support set representations
            support_labels: (N,) - support set labels
            selection_weights: (N,) - optional weights from adaptive selection
        Returns:
            prototypes: (num_classes, feature_dim) - class prototypes
            class_counts: (num_classes,) - number of samples per class
        """
        device = support_representations.device
        prototypes = torch.zeros(self.num_classes, self.feature_dim, device=device)
        class_counts = torch.zeros(self.num_classes, device=device)
        
        for c in range(self.num_classes):
            # Find samples belonging to class c
            class_mask = (support_labels == c).float()
            
            if selection_weights is not None:
                # Weight by selection probabilities
                class_mask = class_mask * selection_weights
            
            class_count = class_mask.sum()
            
            if class_count > 0:
                # Compute weighted average
                weighted_sum = torch.sum(
                    support_representations * class_mask.unsqueeze(1),
                    dim=0
                )
                prototypes[c] = weighted_sum / class_count
                class_counts[c] = class_count
            else:
                # Use fallback prototype if class not present
                if self.fallback_initialized:
                    prototypes[c] = self.fallback_prototypes[c]
                else:
                    # If fallback not initialized, use mean of all support
                    prototypes[c] = support_representations.mean(dim=0)
        
        return prototypes, class_counts
    
    def update_fallback_prototypes(self, all_representations, all_labels):
        """
        Update fallback prototypes using entire training set
        Call this once at the beginning of training/testing
        
        Args:
            all_representations: (M, feature_dim) - all training representations
            all_labels: (M,) - all training labels
        """
        device = all_representations.device
        
        for c in range(self.num_classes):
            class_mask = (all_labels == c)
            if class_mask.sum() > 0:
                self.fallback_prototypes[c] = all_representations[class_mask].mean(dim=0)
            else:
                # Extremely rare case: class not in training set at all
                self.fallback_prototypes[c] = all_representations.mean(dim=0)
        
        self.fallback_initialized = True
    
    def generate_posterior(self, prototypes):
        """
        Generate posterior distribution parameters for θ_k
        
        Args:
            prototypes: (num_classes, feature_dim) - class prototypes
        Returns:
            theta_mean: (num_classes, feature_dim) - mean of θ_k
            theta_logvar: (num_classes, feature_dim) - log variance of θ_k
        """
        # Process each class prototype
        processed = self.prototype_processor(prototypes)  # (num_classes, hidden_dim)
        
        # Generate posterior parameters
        theta_mean = self.posterior_mean(processed)      # (num_classes, feature_dim)
        theta_logvar = self.posterior_logvar(processed)  # (num_classes, feature_dim)
        
        return theta_mean, theta_logvar
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick for sampling from posterior
        θ_k = μ + σ ⊙ ε, where ε ~ N(0, I)
        
        Args:
            mean: (num_classes, feature_dim)
            logvar: (num_classes, feature_dim)
        Returns:
            theta: (num_classes, feature_dim) - sampled task-specific parameters
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def predict(self, theta, query_representation):
        """
        Make prediction using task-specific parameters
        ỹ_k = θ_k · z̃_k
        
        Args:
            theta: (num_classes, feature_dim) - task-specific parameters
            query_representation: (feature_dim,) or (batch_size, feature_dim)
        Returns:
            logits: (num_classes,) or (batch_size, num_classes) - prediction logits
        """
        # Handle both single query and batch
        if query_representation.dim() == 1:
            # Single query: (feature_dim,)
            # θ_k: (num_classes, feature_dim)
            # Result: (num_classes,)
            logits = torch.matmul(theta, query_representation)
        else:
            # Batch of queries: (batch_size, feature_dim)
            # θ_k: (num_classes, feature_dim)
            # Result: (batch_size, num_classes)
            logits = torch.matmul(query_representation, theta.T)
        
        return logits
    
    def forward(self, support_representations, support_labels, 
                query_representation, selection_weights=None):
        """
        Complete forward pass: generate θ_k and make prediction
        
        Args:
            support_representations: (N, feature_dim)
            support_labels: (N,)
            query_representation: (feature_dim,) or (batch_size, feature_dim)
            selection_weights: (N,) - optional weights from adaptive selection
        Returns:
            logits: prediction logits
            theta_mean: mean of task-specific parameters (optional)
            theta_logvar: log variance of task-specific parameters (optional)
        """
        # Step 1: Compute class prototypes
        prototypes, class_counts = self.compute_class_prototypes(
            support_representations, support_labels, selection_weights
        )
        
        # Step 2: Generate posterior distribution
        theta_mean, theta_logvar = self.generate_posterior(prototypes)
        
        # Step 3: Sample task-specific parameters
        # Sample multiple times for uncertainty estimation
        logits_samples = []
        for _ in range(self.n_samples):
            theta = self.reparameterize(theta_mean, theta_logvar)
            logits = self.predict(theta, query_representation)
            logits_samples.append(logits)
            
        # Stack and return all samples
        logits = torch.stack(logits_samples).mean(dim=0)  # (num_classes,) or (batch_size, num_classes)
        return logits, theta_mean, theta_logvar