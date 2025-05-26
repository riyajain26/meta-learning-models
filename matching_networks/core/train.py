import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def matching_network_train(model, train_meta_dataset, device, epochs=5, lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    """
    Train the Matching Network model on a meta-learning dataset.

    Args:
        model: MatchingNetwork model instance
        train_meta_dataset: a meta-learning dataset object that yields (support_x, support_y, query_x, query_y)
        epochs: number of training epochs
        lr: learning rate
        loss_fn: loss function (default: CrossEntropyLoss)
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_tasks = len(train_meta_dataset)
    
        for task in train_meta_dataset:
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

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Epoch statistics
        avg_loss = total_loss / total_tasks
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
