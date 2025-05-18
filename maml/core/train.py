import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call


## Meta-Learning training function using MAML
## This function performs both:
## -Inner-loop: task-specific adaptation/fine-tuning and 
## -Outer-loop: meta-update to generalize across tasks

def maml_train(model, train_meta_dataset, device, epochs=5, inner_steps=1, inner_lr=0.01, meta_lr=0.001, loss_fn=nn.CrossEntropyLoss()):
    # Meta-optimizer: optimizes model's base parameters based on query set loss
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    model.to(device)

    for epoch in range(epochs):
        # Regenerate tasks for this epoch (ensures task diversity)
        train_meta_dataset.tasks = train_meta_dataset.create_tasks()

        # Reset gradients of meta-optimizer
        meta_optimizer.zero_grad()

        # Accumulate loss across all tasks for this meta-update
        meta_loss = 0.0
        total_tasks = len(train_meta_dataset)
        
        # Iterate through each meta-task
        for task in train_meta_dataset:
            # Get the support and query sets for the task
            support_x, support_y, query_x, query_y = task

            # Move data to the appropriate device (CPU/GPU)
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # Get a copy of current model parameters (initial parameters - used for task-specific adaptation)
            params = dict(model.named_parameters())    

            # ---------------------- INNER LOOP ----------------------
            # Perform task-specific adaptation using the support set
            for _ in range(inner_steps):
                # Forward pass on support set with current params
                support_preds = functional_call(model, params, (support_x,))
                inner_loss = loss_fn(support_preds, support_y)

                # Compute gradients w.r.t. current params
                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=True)

                # Manually update the parameters with gradient descent
                params = {
                    name: param - (inner_lr * grad) 
                    for (name,param), grad in zip(params.items(), grads)
                }
            # ---------------------- END INNER LOOP ----------------------

            # ---------------------- OUTER LOOP ----------------------
            # Forward pass/evaluation on query set using adapted parameters
            query_preds = functional_call(model, params, (query_x,))
            query_loss = loss_fn(query_preds, query_y)

            # Accumulate the loss for meta-update
            meta_loss += query_loss
            # ---------------------- END OUTER LOOP ----------------------

        # Meta-update
        # Compute average meta-loss across tasks
        meta_loss /= total_tasks

        # Backpropagate through the meta-loss and update base model parameters
        meta_loss.backward()
        meta_optimizer.step()

        # Logging for progress tracking
        print(f"Epoch {epoch+1}/{epochs}, Train Meta Loss: {meta_loss.item():.4f}")