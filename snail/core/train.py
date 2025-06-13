import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def snail_train(model, train_meta_dataset, device, epochs=5, lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    """
    Trains the SNAIL model on a meta-learning dataset.

    Args:
        model (nn.Module): The SNAILWrapper model (encoder + SNAIL).
        train_meta_dataset (iterable): Meta-dataset yielding tasks.
        device (torch.device): Device to train on (e.g., cuda or cpu).
        epochs (int): Number of meta-training epochs.
        lr (float): Learning rate.
        loss_fn: Loss function to use (default: CrossEntropyLoss).

    Each task contains:
        - support_x: (K, C, L)
        - support_y: (K,)
        - query_x:   (Q, C, L)
        - query_y:   (Q,)
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    # Total number of classes per task
    num_classes = train_meta_dataset.num_classes

    for epoch in range(epochs):
        total_loss = 0.0
        total_tasks = len(train_meta_dataset)

        for task in train_meta_dataset:
            # Unpack the meta-task: support and query sets: K = N_way*K_shot, Q = N_way*Q_query, and L = input_time length
            support_x, support_y, query_x, query_y = task   # Shapes: (K, C, L) and (K) and (Q, C, L) and (Q)

            # Combine support and query inputs → shape: (K + Q, C, L)
            eeg_x = torch.cat([support_x, query_x], dim=0).to(device)

            # One-hot encode the labels for the support and query sets
            support_y = F.one_hot(support_y, num_classes=num_classes).float()
            query_y = F.one_hot(query_y, num_classes=num_classes).float()

            # Create label tensor where query labels are zeroed out
            # This prevents the model from seeing the answers for the query set
            query_y_0 = torch.zeros_like(query_y)
            eeg_y = torch.cat([support_y, query_y_0], dim=0).to(device)

            # Forward pass through SNAIL model
            # eeg_x: (K+Q, C, L), eeg_y: (K+Q, num_classes)
            logits = model(eeg_x, eeg_y)    # Output: (K+Q, num_classes)

            # Isolate the predictions for the query set
            T_support = support_x.size()[0]     # Number of support examples
            query_logits = logits[T_support:]   # Only evaluate query predictions

            # Convert query_y to class indices for loss computation
            query_y = torch.argmax(query_y, dim=1)   # Shape: (Q,)

            # Compute loss on query set predictions
            loss = loss_fn(query_logits, query_y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Epoch statistics
        avg_loss = total_loss / total_tasks
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")