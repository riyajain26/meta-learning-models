import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SharedEncoder(nn.Module):
    """
    Encoder shared for Time- and Frequency-representation
    """
    def __init__(self, in_channels=22, time_length=1875, hidden_dim=128):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (batch, hidden_dim, 1)

    def forward(self, x):
        # x = (batch, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return x


class TimeFrequencyVAE(nn.Module):
    """
    Time- and Frequency-enhanced Representation Learning Module
    Uses dual VAEs to learn disentangled time and frequency representation
    Shared Encoder applied to time- and frequency-domain signals
    """

    def __init__(self, in_channels=22, time_length=1875, num_classes=4, latent_dim=64, hidden_dim=128):
        """
        Args:
            in_channels: Number of EEG channels (22)
            time_length: Length of time series (1875)
            num_classes: Number of classes (4)
            latent_dim: Dimension of latent representation (z_T and z_F each)
            hidden_dim: Hidden layer dimension
        """
        super(TimeFrequencyVAE, self).__init__()

        self.in_channels = in_channels
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.feature_dim = latent_dim * 2           #Concatenate z_T and z_F
        self.freq_bins = (time_length//2) + 1       #rfft bins

        # Shared Encoder for both Time- and Frequency-domain
        self.shared_encoder = SharedEncoder(in_channels, time_length, hidden_dim) # (B, hidden_dim)

        # Time domain encoder branches (mean and log_var)
        self.time_encoder_mean = nn.Linear(hidden_dim, latent_dim)      # (B, latent_dim)
        self.time_encoder_logvar = nn.Linear(hidden_dim, latent_dim)    # (B, latent_dim)

        # Frequency domain encoder branches (mean and log_var)
        self.freq_encoder_mean = nn.Linear(hidden_dim, latent_dim)      # (B, latent_dim)
        self.freq_encoder_logvar = nn.Linear(hidden_dim, latent_dim)    # (B, latent_dim)

        # Time domain decoder
        self.time_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels * time_length) # output raw waveform
        )
        
        # Frequency domain decoder
        self.freq_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels * self.freq_bins)
        )

        # For mutual information estimation (simplified using critics)
        self.time_label_critic = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.freq_label_critic = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.time_freq_critic = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        """
        Encode input into time and frequency representations
        Args:
            x: (batch_size, n_channels, time_length)
        Returns:
            z_T_mean, z_T_logvar: Time domain latent distribution parameters
            z_F_mean, z_F_logvar: Frequency domain latent distribution parameters
        """
        # Shared encoding
        shared_features = self.shared_encoder(x)  # (B, hidden_dim)
        
        # Time domain encoding
        z_T_mean = self.time_encoder_mean(shared_features)      # (B, latent_dim)
        z_T_logvar = self.time_encoder_logvar(shared_features)  # (B, latent_dim)
        
        # Frequency domain encoding
        z_F_mean = self.freq_encoder_mean(shared_features)      # (B, latent_dim)
        z_F_logvar = self.freq_encoder_logvar(shared_features)  # (B, latent_dim)         
        
        return z_T_mean, z_T_logvar, z_F_mean, z_F_logvar
    
    def decode(self, z_T, z_F):
        """
        Decode time and frequency representations
        Args:
            z_T: Time domain latent (batch_size, latent_dim)
            z_F: Frequency domain latent (batch_size, latent_dim)
        Returns:
            x_T_recon: Time domain reconstruction
            x_F_recon: Frequency domain reconstruction
        """
        # Time domain decoding
        x_T_recon = self.time_decoder(z_T)      # (B, in_channels * time_length)
        x_T_recon = x_T_recon.view(-1, self.in_channels, self.time_length)
        
        # Frequency domain decoding
        x_F_recon = self.freq_decoder(z_F)      # (B, in_channels * freq_bins)
        x_F_recon = x_F_recon.view(-1, self.in_channels, self.freq_bins)
        
        return x_T_recon, x_F_recon
    
    def forward(self, x, labels=None):
        """
        Forward pass
        Args:
            x: (batch_size, n_channels, time_length)
            labels: (batch_size,) - optional, for mutual information calculation
        Returns:
            z: Concatenated representation (batch_size, latent_dim*2)
            reconstruction losses and KL divergences
        """
        batch_size = x.size(0)
        
        # Encode
        z_T_mean, z_T_logvar, z_F_mean, z_F_logvar = self.encode(x)     # (B/N, latent_dim) : all
        
        # Sample latent variables
        z_T = self.reparameterize(z_T_mean, z_T_logvar)                 # (B/N, latent_dim)    
        z_F = self.reparameterize(z_F_mean, z_F_logvar)                 # (B/N, latent_dim)
        
        # Decode
        x_T_recon, x_F_recon = self.decode(z_T, z_F)                    # (B/N, in_channels, time_length) and (B/N, in_channels, freq_bins)
        
        # Concatenate for final representation
        z = torch.cat([z_T, z_F], dim=1)  # (batch_size, latent_dim*2)
        
        # Compute losses
        losses = {}

        # Time domain reconstruction loss
        time_recon_loss = F.mse_loss(x_T_recon, x, reduction='mean')
        losses['time_recon'] = time_recon_loss
        
        # Frequency domain reconstruction loss
        # Convert to frequency domain using FFT
        # Use amplitude for reconstruction
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_amp = torch.abs(x_freq)
        x_F_recon_amp = torch.abs(x_F_recon)
        freq_recon_loss = F.mse_loss(x_F_recon_amp, x_freq_amp, reduction='mean')
        losses['freq_recon'] = freq_recon_loss

        # KL divergence for time domain
        kl_time = -0.5 * torch.sum(1 + z_T_logvar - z_T_mean.pow(2) - z_T_logvar.exp())
        kl_time = kl_time / batch_size
        losses['kl_time'] = kl_time
        
        # KL divergence for frequency domain
        kl_freq = -0.5 * torch.sum(1 + z_F_logvar - z_F_mean.pow(2) - z_F_logvar.exp())
        kl_freq = kl_freq / batch_size
        losses['kl_freq'] = kl_freq

        # Mutual information estimation (if labels provided)
        if labels is not None:
            # Convert labels to one-hot
            labels_onehot = F.one_hot(labels, self.num_classes).float()
            
            # I(y, z_T) - maximize
            time_label_joint = torch.cat([z_T, labels_onehot], dim=1)
            mi_time_label = self.time_label_critic(time_label_joint).mean()
            losses['mi_time_label'] = -mi_time_label  # Negative because we maximize
            
            # I(y, z_F) - maximize
            freq_label_joint = torch.cat([z_F, labels_onehot], dim=1)
            mi_freq_label = self.freq_label_critic(freq_label_joint).mean()
            losses['mi_freq_label'] = -mi_freq_label
            
            # I(z_T, z_F) - minimize
            time_freq_joint = torch.cat([z_T, z_F], dim=1)
            mi_time_freq = self.time_freq_critic(time_freq_joint).mean()
            losses['mi_time_freq'] = mi_time_freq  # Positive because we minimize

        return z, losses


    def get_representation(self, x):
        """
        Get representation without computing losses (for inference)
        Args:
            x: (batch_size, n_channels, time_length)
        Returns:
            z: (batch_size, latent_dim*2)
        """
        with torch.no_grad():
            z_T_mean, _, z_F_mean, _ = self.encode(x)
            z = torch.cat([z_T_mean, z_F_mean], dim=1)
        return z
