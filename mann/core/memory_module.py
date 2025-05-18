import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    """
    Memory Module for Differentiable Memory Networks.

    This class initializes and manages an external memory matrix that is 
    read from and written to by read and write heads.

    Args:
        memory_size (int): Number of memory slots (N).
        word_size (int): Size of each memory word (W).
    """
    def __init__(self, memory_size, word_size):
        super().__init__()
        self.N = memory_size
        self.W = word_size
        self.memory = None

    def reset(self, batch_size, device):
        """
        Initializes/reset the memory to zeros for a given batch size.

        Args:
            batch_size (int): Number of samples in the batch.
            device (torch.device): Device on which memory should be allocated.
        """
        self.memory = torch.zeros(batch_size, self.N, self.W, device=device)    ##Shape: (B, N, W)

    def get_memory(self):
        """
        Returns the current memory state.

        Returns:
            Tensor: Memory tensor of shape (B, N, W)
        """
        return self.memory

    def set_memory(self, new_memory):
        """
        Sets the memory to a new value.

        Args:
            new_memory (Tensor): Updated memory tensor of shape (B, N, W)
        """
        self.memory = new_memory



class ReadHead(nn.Module):
    """
    Read Head for reading from the memory using cosine similarity.

    Args:
        memory (Memory): Shared memory object.
        num_read_heads (int): Number of read heads.
    """
    def __init__(self, memory, num_read_heads):
        super().__init__()
        self.memory = memory
        self.num_read_heads = num_read_heads
        
    def forward(self, read_keys):
        """
        Perform read operation using cosine similarity between keys and memory.

        Args:
            read_keys (Tensor): Read keys of shape (B, n, W)

        Returns:
            Tuple[Tensor, Tensor]:
                - read_vectors: Read vectors of shape (B, n, W)
                - read_weights: Read weights (attention) of shape (B, n, N)
        """

        B, n, W = read_keys.shape
        
        # Get the current memory state
        mem = self.memory.get_memory()          # Shape: (B, N, W)   
        mem_norm = F.normalize(mem, dim=2)      # Shape: (B, N, W)
        k_norm = F.normalize(read_keys, dim=2)  # Shape: (B, n, W)

        # Cosine similarity between normalized keys and memory
        K_kM = torch.bmm(k_norm, mem_norm.transpose(1,2))   # Shape: (B, n, N)
        read_weights = F.softmax(K_kM, dim=2)                # Shape: (B, n, N)
        
        # Weighted sum over memory slots using attention weights
        read_vectors = torch.bmm(read_weights, mem)           # Shape: (B, n, N) ** (B, N, W) -> (B, n, W)

        return read_vectors, read_weights
    


class WriteHead(nn.Module):
    """
    Write Head for writing into the memory using Least Recently Used (LRU) strategy.

    Args:
        memory (Memory): Shared memory object.
    """
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.sigmoid = nn.Sigmoid()

    def forward(self, write_key, prev_read_weights, prev_usage_weights, alpha):
        """
        Perform write operation based on usage weights and content similarity.

        Args:
            write_key (Tensor): Key to be written, shape (B, W)
            prev_read_weights (Tensor): Previous read weights, shape (B, n, N)
            prev_usage_weights (Tensor): Previous usage vector, shape (B, N)
            alpha (Tensor): Interpolation gate between read-based and LRU-based write, shape (B, 1)

        Returns:
            Tensor: Final write weights of shape (B, N)
        """

        B, W = write_key.shape
        mem = self.memory.get_memory()

        # LRU: Identify least-used memory slot (lowest usage weight)
        _, indices = torch.topk(prev_usage_weights, k=1, largest=False)     # Shape: (B, 1)
        prev_write_weights_lru = F.one_hot(indices.squeeze(1), num_classes=self.memory.N).float()    # Shape: (B, N)

        # Compute gated write weights between read weights and LRU weights
        sigma_alpha = self.sigmoid(alpha)       # Shape: (B, 1)
        write_weights = sigma_alpha*prev_read_weights[:,0] + (1-sigma_alpha)*prev_write_weights_lru  # Shape: (B, N)

        # Prepare for memory update
        write_weights = write_weights.unsqueeze(2)      # Shape: (B, N, 1)
        add_vector = write_key.unsqueeze(1)             # Shape: (B, 1, W)

        # Memory update: weighted addition
        new_mem = mem + (write_weights * add_vector)
        self.memory.set_memory(new_mem)
        
        return write_weights.squeeze(2)     # Return shape: (B, N)