import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Meta-testing function using REPTILE
# Evaluates the meta-learned model on unseen tasks (i.e., new participants)
# Only the inner-loop (task-specific adaptation) is performed; no meta-updates here

def reptile_test(model, test_meta_dataset, device, inner_steps=1, inner_lr=0.01, loss_fn=nn.CrossEntropyLoss()):
    model.to(device)
    model.eval()

    total_tasks = len(test_meta_dataset)
        
    meta_loss = 0.0

    # Iterate through each meta-task
    for task in test_meta_dataset:
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

    # Average loss across all test tasks
    avg_test_loss = meta_loss / total_tasks
    
    print(f"Test Meta Loss (Unseen task): {avg_test_loss.item():.4f}")