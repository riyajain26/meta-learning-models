import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def matching_network_train(model, train_meta_dataset, device, epochs=5, lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    """
    Trains the Matching Network on a meta-learning dataset using episodic training.

    Args:
        model: Matching Network model instance.
        train_meta_dataset: Meta-learning dataset that yields tasks in the form 
                            (support_x, support_y, query_x, query_y).
        device: Device on which training will run (e.g., 'cuda' or 'cpu').
        epochs: Number of meta-epochs (i.e., how many times to iterate over meta-tasks).
        lr: Learning rate for the optimizer.
        loss_fn: Loss function to use (default: CrossEntropyLoss).
    """

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Move model to computation device and set to training mode
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_tasks = len(train_meta_dataset)       # Total number of meta-tasks in this epoch
    
        # Iterate through each meta-task (an episode)
        for task in train_meta_dataset:
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

            # === Backward Pass and Optimization ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for monitoring
            total_loss += loss.item()

        # Average loss over all meta-tasks in this epoch
        avg_loss = total_loss / total_tasks
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
