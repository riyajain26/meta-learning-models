import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def matching_network_test(model, test_meta_dataset, device, loss_fn=nn.CrossEntropyLoss()):
    """
    Evaluate the Matching Network model on a meta-test dataset.

    Args:
        model: Trained meta-learning model with a Matching Network architecture.
        test_meta_dataset: Meta-test dataset (few-shot task episodes).
        loss_fn: Loss function (default: CrossEntropyLoss).

    Returns:
        Tuple of (average loss, average accuracy) across all test tasks.
    """

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tasks = len(test_meta_dataset)

    with torch.no_grad():
        for task in test_meta_dataset:
            # Unpack the meta-task: support and query sets: K = N_way*K_shot, Q = N_way*Q_query, and L = input_time length
            support_x, support_y, query_x, query_y = task   # Shapes: (K, C, L) and (K) and (Q, C, L) and (Q)

            # Reshape and send data to the device - setting batch = 1 
            support_x = support_x.to(device)        # Shape: (K, C, L)
            support_y = support_y.to(device)        # Shape: (K,)
            #support_y = F.one_hot(support_y, num_classes=num_classes).float()   # Shape: (K, num_classes)
            
            query_x = query_x.to(device)            # Shape: (Q, C, L)
            query_y = query_y.to(device)            # Shape: (Q,)
            #query_y = F.one_hot(query_y, num_classes=num_classes).float()       # Shape: (Q, num_classes)
            
            # Forward pass through the wrapper model
            logits = model(support_x, support_y, query_x)      # Shape: (T=K+Q, num_classes)

            # Compute loss
            loss = loss_fn(logits, query_y)

            total_loss += loss.item()

    # Epoch statistics
    avg_loss = total_loss / total_tasks
        
    print(f"Test Loss: {avg_loss:.4f}")
    
    # You could add more detailed analysis here (confusion matrix, per-class metrics, etc.)
    return avg_loss
