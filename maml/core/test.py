import torch
import torch.nn as nn
from torch.func import functional_call

# Meta-testing function using MAML
# Evaluates the meta-learned model on unseen tasks (i.e., new participants)
# Only the inner-loop (task-specific adaptation) is performed; no meta-updates here

def maml_test(model, test_meta_dataset, device, inner_steps=1, inner_lr=0.01, loss_fn=nn.CrossEntropyLoss()):
    model.to(device)
    model.eval()        # Set model to evaluation mode  

    meta_loss = 0.0     # Total loss across all unseen tasks
    total_tasks = len(test_meta_dataset)

    for task in test_meta_dataset:
        # Get support and query sets for the current test task
        support_x, support_y, query_x, query_y = task

        # Move tensors to device (GPU/CPU)
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        # Copy of current model parameters for task-specific adaptation
        params = dict(model.named_parameters())

        # ---------------------- INNER LOOP ----------------------
        # Adapt model parameters on the support set of the unseen task
        for _ in range(inner_steps):
            # Forward pass on support set using current task-specific parameters
            support_preds = functional_call(model, params, (support_x,))
            inner_loss = loss_fn(support_preds, support_y)

            # Compute gradients with respect to the parameters
            grads = torch.autograd.grad(inner_loss, params.values())

            # Manually update parameters using gradient descent
            params = {
                name: param - (inner_lr * grad) 
                for (name,param), grad in zip(params.items(), grads)
            }
        # ---------------------- END INNER LOOP ----------------------

        # ---------------------- EVALUATION ----------------------
        # Evaluate adapted model on query set (simulate generalization)
        query_preds = functional_call(model, params, (query_x,))
        query_loss = loss_fn(query_preds, query_y)

        # Accumulate total meta-test loss
        meta_loss += query_loss
        # ---------------------- END EVALUATION ----------------------

    # Average loss across all test tasks
    avg_test_loss = meta_loss / total_tasks
    
    print(f"Test Meta Loss (Unseen task): {avg_test_loss.item():.4f}")