import torch
import torch.nn as nn

class LSTMController(nn.Module):
    """
    LSTM-based Controller Module for Differentiable Memory Networks.

    This controller is typically used in Neural Turing Machines (NTMs), Differentiable Neural Computers (DNCs),
    or Memory-Augmented Neural Networks (MANNs). It takes an input vector and internal state, then outputs:
      - a control output (e.g., logits for classification),
      - updated hidden and cell states,
      - key vectors used for memory read/write operations.

    Args:
        input_size (int): Size of the input feature vector at each time step.
        hidden_size (int): Number of hidden units in the LSTM cell.
        num_read_heads (int): Number of memory read heads (used in memory addressing).
        word_size (int): Size of the memory word vector.
        output_size (int): Size of the final output (e.g., number of classes).
    """
    def __init__(self, input_size, hidden_size, num_read_heads, word_size, output_size):
        super().__init__()

        # LSTM cell for processing input and maintaining internal state
        # Input:  (B, input_size)
        # Output: hidden state h of shape (B, hidden_size)
        self.lstm = nn.LSTMCell(input_size, hidden_size)    

        # Linear layer to produce memory keys (for read and write heads)
        # Output shape: (B, (num_read_heads + 1) * word_size)
        # One extra head is used for write key
        self.key_layer = nn.Linear(hidden_size, (num_read_heads + 1) * word_size)

        # Output layer to generate final output (e.g., logits for classification)
        # Output shape: (B, output_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_state):
        """
        Forward pass of the LSTMController.

        Args:
            x (Tensor): Input tensor at current time step, shape (B, input_size)
            prev_state (Tuple[Tensor, Tensor]): Tuple of previous hidden and cell states (h_prev, c_prev),
                                                each of shape (B, hidden_size)

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
                - out: Output tensor of shape (B, output_size)
                - (h, c): Updated hidden and cell states
                - keys: Tensor of memory keys for read/write, shape (B, (num_read_heads + 1) * word_size)
        """

        # Unpack previous state
        h_prev, c_prev = prev_state

        # Update internal state using LSTM cell
        h, c = self.lstm(x, (h_prev, c_prev))  

        # Compute memory keys (used by read and write heads)
        keys = self.key_layer(h)

        # Compute final output (e.g., logits)
        out = self.output_layer(h)
        
        return out, (h, c), keys