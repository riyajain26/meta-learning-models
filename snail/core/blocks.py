import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Causal 1D Convolution Layer ---
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()

        # Calculate padding to ensure causality (no future leakage)
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, filters, kernel_size,
                         padding=self.padding, dilation=dilation)

    def forward(self, x):
        # Input: (B, T, C), expected by Conv1d as (B, C, T), so this assumes input is already transposed
        out = self.conv1d(x)

        # Remove extra right padding to maintain causal structure
        if self.padding != 0:                   
            out = out[:, :, :-self.padding]     
        
        return out

   
# --- Dense Block ---
class DenseBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dilation=1):
        super(DenseBlock, self).__init__()
        # Two causal conv layers for gating mechanism
        self.conv_f = CausalConv1d(in_channels, filters, kernel_size, dilation)
        self.conv_g = CausalConv1d(in_channels, filters, kernel_size, dilation)

    def forward(self, x):
        # f: tanh activation branch
        f = torch.tanh(self.conv_f(x))
        # g: sigmoid activation branch (gate)
        g = torch.sigmoid(self.conv_g(x))
        # Gated activation
        activations = f * g
        # Concatenate input with new features (dense connectivity)
        return torch.cat((x, activations), dim=1)
    

# --- Temporal Convolution Block (multiple DenseBlocks with increasing dilation) ---
class TCBlock(nn.Module):
    def __init__(self, in_channels, time_length, filters, kernel_size=2):
        super().__init__()
        self.dense_blocks = nn.ModuleList()

        # Number of layers based on log2(time_length)
        for i in range(int(math.ceil(math.log(time_length, 2)))):
            # Input to each DenseBlock increases because each previous block concatenates its output to the input.
            block = DenseBlock(
                in_channels + i * filters, 
                filters=filters, 
                kernel_size=kernel_size, 
                dilation=2**(i+1)
            )    
            self.dense_blocks.append(block)

    def forward(self, x):
        # Input: (B, T, C) => Convert to (B, C, T) for Conv1d
        x = torch.transpose(x, 1, 2)
        for block in self.dense_blocks:
            x = block(x)
        # Output: Transpose back to (B, T, C)
        return torch.transpose(x, 1, 2)
    

# --- Attention Block ---
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, val_size):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(in_channels, key_size)
        self.key = nn.Linear(in_channels, key_size)
        self.value = nn.Linear(in_channels, val_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, x):
        # Input shape: (B, T, C)
        queries = self.query(x)  # (B, T, key_size)
        keys = self.key(x)       # (B, T, key_size)
        values = self.value(x)   # (B, T, val_size)

        # Compute scaled dot-product attention scores
        logits = torch.bmm(queries, keys.transpose(1, 2))   # (B, T, T)

        # Apply causal mask: prevent attending to future time steps
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)   # (1, T, T)
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # Softmax over time dimension (causal attention)
        probs = torch.softmax(logits/self.sqrt_key_size, dim=1)     # (B, T, T)
        read = torch.bmm(probs, values)     # (B, T, val_size)

        # Concatenate input and attention output
        return torch.cat((x, read), dim=2)  # (B, T, C + val_size)