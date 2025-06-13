import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def snail_test(model, test_meta_dataset, device, loss_fn=nn.CrossEntropyLoss()):
    """
    Evaluates the SNAIL model on a meta-testing dataset.

    Args:
        model (nn.Module): The trained SNAILWrapper model (encoder + SNAIL).
        test_meta_dataset (iterable): Meta-dataset yielding test tasks.
        device (torch.device): Device to evaluate on (e.g., 'cuda' or 'cpu').
        loss_fn: Loss function used to compute test loss.

    Each task contains:
        - support_x: (K, C, L)
        - support_y: (K,)
        - query_x:   (Q, C, L)
        - query_y:   (Q,)
    """

    model.to(device)
    model.eval()

    # Total number of classes per task
    num_classes = test_meta_dataset.num_classes
    total_loss = 0.0
    total_tasks = len(test_meta_dataset)

    with torch.no_grad():       # Disable gradients for evaluation
        for task in test_meta_dataset:
            # Unpack the meta-task: support and query sets: K = N_way*K_shot, Q = N_way*Q_query, and L = input_time length
            support_x, support_y, query_x, query_y = task   # Shapes: (K, C, L) and (K) and (Q, C, L) and (Q)

            # Combine support and query inputs → shape: (K + Q, C, L)
            eeg_x = torch.cat([support_x, query_x], dim=0).to(device)

            # One-hot encode labels
            support_y = F.one_hot(support_y, num_classes=num_classes).float()
            query_y = F.one_hot(query_y, num_classes=num_classes).float()

            # Create label tensor where query labels are zeroed out
            # This prevents the model from seeing the answers for the query set
            query_y_0 = torch.zeros_like(query_y)
            eeg_y = torch.cat([support_y, query_y_0], dim=0).to(device)

            # Forward pass through the SNAIL model
            logits = model(eeg_x, eeg_y)  # Output shape: (K+Q, num_classes)

            # Isolate the predictions for the query set
            T_support = support_x.size()[0]     # Number of support examples
            query_logits = logits[T_support:]   # Only evaluate query predictions

            # Convert query_y to class indices for loss computation
            query_y = torch.argmax(query_y, dim=1)   # Shape: (Q,)

            # Compute loss on query set
            loss = loss_fn(query_logits, query_y)
            total_loss += loss.item()

    # Compute average loss over all test tasks
    avg_loss = total_loss / total_tasks
    
    print(f"Test Loss: {avg_loss:.4f}")
    
    # You could add more detailed analysis here (confusion matrix, per-class metrics, etc.)
    return avg_loss