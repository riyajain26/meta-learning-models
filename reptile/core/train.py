import torch
import torch.nn as nn
import torch.optim as optim
import copy

## Meta-Learning training function using REPTILE model
## This function performs both:
## -Inner-loop: task-specific adaptation/fine-tuning and 
## -Outer-loop: meta-update to generalize across tasks

def reptile_train(model, train_meta_dataset, device, epochs=5, inner_steps=1, inner_lr=0.01, meta_lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    model.to(device)

    for epoch in range(epochs):
        # Regenerate tasks for this epoch (ensures task diversity)
        train_meta_dataset.tasks = train_meta_dataset.create_tasks()
        total_tasks = len(train_meta_dataset)

        # Store original weights
        original_weights = copy.deepcopy(model.state_dict())

        # Initialize accumulator for meta-update
        meta_update = {name: torch.zeros_like(param) for name,param in model.state_dict().items()}
        
        meta_loss = 0.0

        # Iterate through each meta-task
        for task in train_meta_dataset:
            # Get the support and query sets for the task
            support_x, support_y, query_x, query_y = task

            # Move data to the appropriate device (CPU/GPU)
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # Create a copy of the model for inner-loop adaptation
            task_model = copy.deepcopy(model)    
            task_model.to(device)
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            # ---------------------- INNER LOOP ----------------------
            for _ in range(inner_steps):
                # Forward pass
                support_preds = task_model(support_x)
                inner_loss = loss_fn(support_preds, support_y)

                # Compute gradients w.r.t. current params
                task_optimizer.zero_grad()
                inner_loss.backward()
                task_optimizer.step()
            # ---------------------- END INNER LOOP ----------------------

            with torch.no_grad():
                query_preds = task_model(query_x)
                query_loss = loss_fn(query_preds, query_y)
                meta_loss += query_loss

            # ---------------------- META UPDATE ----------------------
            # Accumulate difference between adapted and original weights
            adapted_weights = task_model.state_dict()
            for name in meta_update:
                meta_update[name] += adapted_weights[name] - original_weights[name]
            

        # Meta-update      
        for name in model.state_dict():
            updated_param = original_weights[name] + meta_lr * (meta_update[name]/total_tasks)
            model.state_dict()[name].copy_(updated_param)
        # ---------------------- END META UPDATE ----------------------

        # Compute average meta-loss across tasks
        meta_loss /= total_tasks

        # Logging for progress tracking
        print(f"Epoch {epoch+1}/{epochs}, Train Meta Loss: {meta_loss.item():.4f}")