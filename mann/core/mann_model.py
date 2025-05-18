import torch 
import torch.nn as nn
import torch.nn.functional as F

from mann.core.memory_module import Memory, ReadHead, WriteHead
from mann.core.lstm_controller import LSTMController

class MANN(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, word_size, output_size, num_read_heads, device):
        super().__init__()

        # Initialize memory module with given size and word dimensions
        self.memory = Memory(memory_size, word_size)

        # Initialize controller (LSTM) which takes as input the data + memory read vectors
        self.controller = LSTMController(
            input_size + (num_read_heads * word_size), 
            hidden_size, 
            num_read_heads, 
            word_size, 
            output_size
        )

        # Initialize memory heads
        self.read_head = ReadHead(self.memory, num_read_heads)  # reads from memory
        self.write_head = WriteHead(self.memory)                # writes to memory

        # Store hyperparameters
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        # Learnable parameters to control memory usage dynamics
        self.gamma = nn.Parameter(torch.tensor(0.95))       # controls decay of usage weights
        self.alpha = nn.Parameter(torch.tensor(0.5))        # controls interpolation for write weights

    def reset(self, batch_size):
        """
        Reset internal memory and states for new sequence.
        Returns zero-initialized LSTM hidden states and memory usage/read weights.
        """

        self.memory.reset(batch_size, device=self.device)

        h = torch.zeros(batch_size, self.hidden_size, device=self.device)   # LSTM hidden state
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)   # LSTM cell state

        usage_weights = torch.zeros(batch_size, self.memory.N, device=self.device)  # memory usage tracker
        read_weights = torch.zeros(batch_size, self.num_read_heads, self.memory.N, device=self.device)  # read attention weights

        return h, c, usage_weights, read_weights

    def forward(self, x_seq):  
        """
        Process input sequence through MANN.

        Args:
            x_seq: (T, B, input_size) input sequence with time steps T, batch size B

        Returns:
            Output sequence of logits: (T, B, output_size)
        """

        T, B, _ = x_seq.size()
        
        # Initialize states and buffers
        h, c, usage_weights, read_weights = self.reset(B)
        outputs = []
        read_vector = torch.zeros(B, self.num_read_heads * self.word_size, device=self.device)  # initial read vector

        for t in range(T):
            # Combine current input with previous read vector
            x = torch.cat([x_seq[t], read_vector], dim=1)

            # Process through controller (LSTM)
            controller_output, (h, c), keys = self.controller(x, (h, c))

            # Split keys into read and write keys
            keys = keys.view(B, self.num_read_heads+1, self.word_size)
            read_keys = keys[:, :self.num_read_heads]       # for read heads
            write_key = keys[:, -1]                         # for write head

            # Store previous read_weights and usage_weights for write_weight updates 
            prev_read_weights = read_weights
            prev_usage_weights = usage_weights

            # Perform read from memory
            read_vector_t, read_weights = self.read_head(read_keys)     # Shape: (B, n, W) and (B, n, N)

            # Perform write to memory
            write_weights = self.write_head(write_key, prev_read_weights, prev_usage_weights, self.alpha)

            # Update memory usage weights with decay + read + write influence
            usage_weights = (self.gamma * usage_weights) + torch.sum(read_weights, dim=1) + write_weights

            # Flatten the read vector for next step and append controller output
            read_vector = read_vector_t.view(B, -1)
            outputs.append(controller_output)

        # Stack the outputs across time
        return torch.stack(outputs, dim=0)
    

class MANNWrapper(nn.Module):
    def __init__(self, mann, eeg_encoder, num_classes, device):
        super().__init__()
        self.mann = mann
        self.eeg_encoder = eeg_encoder
        self.num_classes = num_classes
        self.device = device

    def forward(self, eeg_seq, label_seq=None):
        """
        Process EEG data and labels through EEG encoder and MANN.

        Args:
            eeg_seq: (T, B, in_channels, input_time) - EEG data over time with default 22 channels and 1875 samples
            label_seq: (T, B, num_classes) - one-hot encoded labels for classification (optional)

        Returns:
            logits: (T, B, num_classes) - predicted logits from MANN
        """

        T,B,C,L = eeg_seq.size()

        # Flatten temporal and batch dimensions for EEG encoder
        eeg_flat = eeg_seq.view(T*B, C, L)

        # Get feature embeddings from EEG encoder
        embeddings = self.eeg_encoder(eeg_flat).view(T,B,-1)    # Shape: (T, B, embedding_dim=128)

        # If no label sequence provided, initialize with zero tensors
        if label_seq is None:
            label_seq = torch.zeros(T,B,self.num_classes, device=self.device)

        # Shift labels for teacher forcing (input at t is label at t-1)
        y_input = torch.zeros_like(label_seq)
        y_input[1:] = label_seq[:-1]    # shift one step forward; shape: (T, B, num_classes)

        # Concatenate EEG embeddings with shifted labels (embeddings + shifted labels)
        mann_input = torch.cat([embeddings, y_input], dim=2)    # shape: (T, B, embedding_dim + num_classes)

        # Forward pass through MANN to get logits
        logits = self.mann(mann_input)  # shape: (T, B, num_classes)
        
        return logits