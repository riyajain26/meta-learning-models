import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_bayesian.core.adaptive_task_constructor import AdaptiveTaskConstructor
from adaptive_bayesian.core.amortized_variational_inference_network import AmortizedVariationalInference
from adaptive_bayesian.core.tfrl_module import TimeFrequencyEncoder

class ABML(nn.Module):
    """
    Complete Adaptive Bayesian Meta-Learning Model
    Integrates all three components
    """

    def __init__(self,
                 in_channels=32,
                 time_length=1044,
                 num_classes=4,
                 latent_dim=64,
                 hidden_dim=128,
                 meta_param_rank=32,
                 temperature_init=1.0,
                 temperature_min=0.1):
        """
        Args:
            n_channels: Number of EEG channels (32)
            time_length: Length of time series (1044)
            n_classes: Number of classes (4)
            latent_dim: Latent dimension for each domain (64)
            hidden_dim: Hidden layer dimension (128)
            meta_param_rank: Rank for meta-parameter factorization (32)
            temperature_init: Initial Gumbel temperature (1.0)
            temperature_min: Minimum Gumbel temperature (0.1)
        """
        super(ABML, self).__init__()
        
        self.n_channels = in_channels
        self.time_length = time_length
        self.num_classes = num_classes
        self.feature_dim = latent_dim * 2  # Concatenated time and frequency
        
        # Component 1: Time-Frequency Encoder
        self.encoder = TimeFrequencyEncoder(
            in_channels=in_channels,
            time_length=time_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        # Component 2: Adaptive Task Constructor
        self.task_constructor = AdaptiveTaskConstructor(
            feature_dim=self.feature_dim,
            meta_param_rank=meta_param_rank,
            temperature_init=temperature_init,
            temperature_min=temperature_min
        )
        
        # Component 3: Amortized Variational Inference
        self.inference_net = AmortizedVariationalInference(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim * 2,
            n_samples=1
        )

    def forward(self, support_pool_data, support_pool_labels,
                query_data, query_labels=None, training=True):
        """
        Complete forward pass
        
        Args:
            support_pool_data: (N, n_channels, time_length)
            support_pool_labels: (N,)
            query_data: (1, n_channels, time_length) or (B, n_channels, time_length)
            query_labels: (1,) or (B,) - optional, for computing losses
            training: bool
            
        Returns:
            logits: (n_classes,) or (B, n_classes)
            losses: dict of all losses
        """
        batch_size = query_data.size(0)
        losses = {}
        
        # Step 1: Extract representations for support pool
        support_pool_representations, encoder_losses_pool = self.encoder(                       # (batch_size, latent_dim*2) : representation; dict : losses
            support_pool_data, support_pool_labels
        )
        # Accumulate pool encoder losses
        for key, value in encoder_losses_pool.items():
            losses[f'pool_{key}'] = value
        
        # Step 2: Extract representations for query
        if query_labels is not None:
            query_representations, encoder_losses_query = self.encoder(
                query_data, query_labels
            )
            # Accumulate encoder losses
            for key, value in encoder_losses_query.items():
                losses[f'query_{key}'] = value
        else:
            query_representations = self.encoder.get_representation(query_data)
        
        # Running for the batch
        all_logits = []
        for batch_idx in range(batch_size):
            query_rep = query_representations[batch_idx]        # (latent_dim*2,) == (feature_dim, )
                
            # Step 3: Adaptive task construction
            selected_data, selected_labels, selected_representations, selection_mask, meta_params = \
                self.task_constructor(
                    support_pool_data, support_pool_labels,
                    support_pool_representations, query_rep,
                    training=training
                )
                
            # Step 4: Amortized inference and prediction
            logits, theta_mean, theta_logvar = self.inference_net(          # (num_classes,)
                selected_representations, selected_labels,
                query_rep, selection_mask if training else None
            )
                
            all_logits.append(logits)
            
        logits = torch.stack(all_logits)
        return logits, losses
            
    
    def compute_loss(self, logits, query_labels, encoder_losses, 
                     alpha=1.0, beta=0.1):
        """
        Compute complete loss function (Equation 10)
        L_ABML = L(ω, φ) + α·L_TFRL
        
        Args:
            logits: (n_classes,) or (B, n_classes)
            query_labels: (1,) or (B,)
            encoder_losses: dict of encoder losses
            alpha: weight for time-frequency representation learning loss
            beta: weight for mutual information terms
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        loss_dict = {}
        
        # Classification loss (negative log-likelihood)
        if logits.dim() == 1:
            # Single query
            classification_loss = F.cross_entropy(logits.unsqueeze(0), query_labels)
        else:
            # Batch of queries
            classification_loss = F.cross_entropy(logits, query_labels)
        
        loss_dict['classification'] = classification_loss
        
        # Time-frequency representation learning losses
        time_recon = encoder_losses.get('query_time_recon', 0) + encoder_losses.get('pool_time_recon', 0)
        freq_recon = encoder_losses.get('query_freq_recon', 0) + encoder_losses.get('pool_freq_recon', 0)
        kl_time = encoder_losses.get('query_kl_time', 0) + encoder_losses.get('pool_kl_time', 0)
        kl_freq = encoder_losses.get('query_kl_freq', 0) + encoder_losses.get('pool_kl_freq', 0)
        
        # ELBO loss
        elbo_loss = time_recon + freq_recon + kl_time + kl_freq
        loss_dict['elbo'] = elbo_loss
        
        # Mutual information losses (if available)
        mi_time_label = encoder_losses.get('query_mi_time_label', 0) + encoder_losses.get('pool_mi_time_label', 0)
        mi_freq_label = encoder_losses.get('query_mi_freq_label', 0) + encoder_losses.get('pool_mi_freq_label', 0)
        mi_time_freq = encoder_losses.get('query_mi_time_freq', 0) + encoder_losses.get('pool_mi_time_freq', 0)
        
        mi_loss = mi_time_label + mi_freq_label - mi_time_freq
        loss_dict['mi'] = mi_loss
        
        # Total time-frequency representation learning loss
        tfrl_loss = elbo_loss + beta * mi_loss
        loss_dict['tfrl'] = tfrl_loss
        
        # Total loss (Equation 10)
        total_loss = classification_loss + alpha * tfrl_loss
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict