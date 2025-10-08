import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ABMLTrainer:
    """Trainer class for ABML model"""
    
    def __init__(self, model, device, learning_rate=1e-3, 
                 alpha=1.0, beta=0.1):
        """
        Args:
            model: ABML model
            device: torch device
            learning_rate: learning rate
            alpha: weight for TFRL loss
            beta: weight for MI loss
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
    def train_epoch(self, train_data, train_labels, support_pool_size=100,
                   batch_size=8):
        """
        Train for one epoch
        
        Args:
            train_data: (N, in_channels, time_length)
            train_labels: (N,)
            support_pool_size: number of samples in support pool
            batch_size: number of query samples per batch
            
        Returns:
            epoch_losses: dict of average losses
        """
        self.model.train()
        
        
        # Sample query batch
        query_indices = torch.randperm(len(train_data))[:batch_size]
        query_data = train_data[query_indices].to(self.device)              ## (B, in_channels, time_length)
        query_labels = train_labels[query_indices].to(self.device)          ## (B, )
            
        # Sample support pool (randomly for each query)
        pool_indices = torch.randperm(len(train_data))[:support_pool_size]
        support_pool_data = train_data[pool_indices].to(self.device)        ## (N, in_channels, time_length)
        support_pool_labels = train_labels[pool_indices].to(self.device)    ## (N, )

        # Initialize fallback prototypes with entire training set
        with torch.no_grad():
            support_pool_representations = self.model.encoder.get_representation(
                support_pool_data
            )
            self.model.inference_net.update_fallback_prototypes(
                support_pool_representations, support_pool_labels
            )
        
        self.optimizer.zero_grad()
        epoch_losses = {}

        # Forward pass
        logits, encoder_losses = self.model(
            support_pool_data, support_pool_labels,
            query_data, query_labels, training=True
        )
            
        # Compute loss
        total_loss, loss_dict = self.model.compute_loss(
            logits, query_labels, encoder_losses,
            alpha=self.alpha, beta=self.beta
        )
            
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
            
        # Accumulate losses
        for key, value in loss_dict.items():
            if key not in epoch_losses:
                epoch_losses[key] = 0
            epoch_losses[key] += value.item()
        
        return epoch_losses
    
    def evaluate(self, train_data, train_labels, test_data, test_labels):
        """
        Evaluate on test set
        
        Args:
            train_data: (N_train, in_channels, time_length) - entire training set
            train_labels: (N_train,)
            test_data: (N_test, in_channels, time_length)
            test_labels: (N_test,)
            
        Returns:
            accuracy: test accuracy
            predictions: predicted labels
        """
        self.model.eval()
        with torch.no_grad():
            train_representations = self.model.encoder.get_representation(
                train_data.to(self.device)
            )
            self.model.inference_net.update_fallback_prototypes(
                train_representations,
                train_labels.to(self.device)
            )
            
        predictions = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                query_data = test_data[i:i+1].to(self.device)
                
                # Use entire training set as support pool during testing
                logits, _ = self.model(
                    train_data.to(self.device), train_labels.to(self.device),
                    query_data, training=False
                )
                
                pred = torch.argmax(logits, dim=1).cpu().item()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        accuracy = (predictions == test_labels.numpy()).mean()
        
        return accuracy, predictions
    
    def train(self, train_data, train_labels, test_data, test_labels,
              n_epochs=100, support_pool_size=100, batch_size=8,
              eval_every=5, save_path=None):
        """
        Complete training loop
        
        Args:
            train_data: training data
            train_labels: training labels
            test_data: test data
            test_labels: test labels
            n_epochs: number of epochs
            support_pool_size: support pool size during training
            batch_size: batch size
            eval_every: evaluate every N epochs
            save_path: path to save best model
            
        Returns:
            history: dict of training history
        """
        history = {
            'train_loss': [],
            'test_accuracy': [],
            'best_accuracy': 0.0
        }
        
        for epoch in range(n_epochs):
            # Update temperature
            self.model.task_constructor.update_temperature(epoch, n_epochs)
            
            # Train
            epoch_losses = self.train_epoch(
                train_data, train_labels,
                support_pool_size=support_pool_size,
                batch_size=batch_size
            )
            
            history['train_loss'].append(epoch_losses['total'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Loss: {epoch_losses['total']:.4f}")
            print(f"  Classification: {epoch_losses['classification']:.4f}")
            print(f"  TFRL: {epoch_losses['tfrl']:.4f}")
            print(f"  Temperature: {self.model.task_constructor.temperature:.3f}")
            
            # Evaluate
            if (epoch + 1) % eval_every == 0:
                accuracy, _ = self.evaluate(
                    train_data, train_labels,
                    test_data, test_labels
                )
                history['test_accuracy'].append(accuracy)
                
                print(f"  Test Accuracy: {accuracy:.4f}")
                
                # Save best model
                if accuracy > history['best_accuracy']:
                    history['best_accuracy'] = accuracy
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  Saved best model to {save_path}")
        
        print(f"Meta-Training Complete!")
        print(f"Best Accuracy: {history['best_accuracy']:.4f}")

        return history