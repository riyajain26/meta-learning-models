import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def matching_network_test(model, test_meta_dataset, device, loss_fn=nn.CrossEntropyLoss()):
    """
    Evaluates the Matching Network model on a set of meta-test tasks.

    Args:
        model (nn.Module): Trained Matching Network model.
        test_meta_dataset (Iterable): Meta-test dataset, where each task is a (support_x, support_y, query_x, query_y) tuple.
        device (torch.device): Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        loss_fn (Callable): Loss function to compute evaluation loss (default: CrossEntropyLoss).

    Returns:
        float: Average loss across all meta-test tasks.
    """

    # Move model to target device and switch to evaluation mode
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tasks = len(test_meta_dataset)        # Total number of tasks for evaluation

    # Disable gradient computation for evaluation (saves memory and computations)
    with torch.no_grad():
        for task in test_meta_dataset:
            # Unpack the meta-task: support and query sets: K = N_way*K_shot, Q = N_way*Q_query, and L = input_time length
            support_x, support_y, query_x, query_y = task   # Shapes: (K, C, L) and (K) and (Q, C, L) and (Q)

            # Move tensors to the correct device
            support_x = support_x.to(device)        # Shape: (K, C, L)
            support_y = support_y.to(device)        # Shape: (K,)
            query_x = query_x.to(device)            # Shape: (Q, C, L)
            query_y = query_y.to(device)            # Shape: (Q,)
            
            # === Forward Pass ===
            # The model takes support set and query set to compute class predictions for queries
            logits = model(support_x, support_y, query_x)      # Shape: (Q, num_classes)

            # === Loss Computation ===
            # CrossEntropyLoss expects logits and target labels (not one-hot)
            loss = loss_fn(logits, query_y)

            # Accumulate total loss
            total_loss += loss.item()

    # Average loss over all meta-tasks
    avg_loss = total_loss / total_tasks
        
    print(f"Test Loss: {avg_loss:.4f}")
    
    # You could add more detailed analysis here (confusion matrix, per-class metrics, etc.)
    return avg_loss
