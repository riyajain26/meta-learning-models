import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def mann_train(model, train_meta_dataset, device, epochs=5, lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    """
    Train the MANN model on a meta-learning dataset.

    Args:
        model: MANNWrapper model instance
        train_meta_dataset: a meta-learning dataset object that yields (support_x, support_y, query_x, query_y)
        epochs: number of training epochs
        lr: learning rate
        loss_fn: loss function (default: CrossEntropyLoss)
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

            # Reshape and send data to the device - setting batch = 1 
            support_x = support_x.unsqueeze(1).to(device)   # Shape: (K, 1, C, L)
            support_y = support_y.unsqueeze(1).to(device)       # Shape: (K, 1)
            support_y = F.one_hot(support_y, num_classes=num_classes).float()   # Shape: (K, 1, num_classes)
            
            query_x = query_x.unsqueeze(1).to(device)       # Shape: (Q, 1, C, L)
            query_y = query_y.unsqueeze(1).to(device)           # Shape: (Q, 1)
            query_y = F.one_hot(query_y, num_classes=num_classes).float()       # Shape: (Q, 1, num_classes)
            
            # Combine support and query into one sequence for MANN
            eeg_seq = torch.cat([support_x, query_x], dim=0)    # Shape: (T=K+Q, 1, C, L)
            label_seq = torch.cat([support_y, query_y], dim=0)  # Shape: (T=K+Q, 1, num_classes)
            
            # Number of support time steps
            T_support = support_x.size(0)

            # Forward pass through the wrapper model
            logits = model(eeg_seq, label_seq)      # Shape: (T=K+Q, B=1, num_classes)
            
            # Slice out the query-time logits for loss computation
            query_logits = logits[T_support:]       # Shape: (Q, 1, num_classes)
            query_logits = query_logits.view(-1, query_logits.size(-1))     # Shape: (Q, num_classes)

            # Ground truth class indices
            query_y = torch.argmax(query_y.view(-1, query_y.size(-1)), dim=1)   # Shape: (Q,)

            # Compute loss
            loss = loss_fn(query_logits, query_y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Epoch statistics
        avg_loss = total_loss / total_tasks
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")