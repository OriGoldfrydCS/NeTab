import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime


class NetworkTrafficTrainer:
    """
    Trainer class for network traffic generation models with comprehensive training features
    
    Features:
    - Supports both frozen base + MLP head training (mode 0) and full fine-tuning (mode 1)
    - Early stopping with patience-based monitoring
    - Learning rate scheduling (cosine annealing or plateau reduction)
    - Gradient clipping for training stability
    - Comprehensive logging and checkpoint management
    - Training history tracking and visualization support
    """
    
    def __init__(
        self,
        model,                                          # NetworkTrafficModel instance
        train_dataloader: DataLoader,                   # Training data loader
        val_dataloader: Optional[DataLoader] = None,    # Validation data loader
        learning_rate: float = 5e-5,                    # Learning rate for optimizer
        weight_decay: float = 0.01,                     # Weight decay for regularization
        warmup_steps: int = 100,                        # Number of warmup steps (unused currently)
        max_grad_norm: float = 1.0,                     # Maximum gradient norm for clipping
        save_dir: str = "./models",                     # Directory to save model checkpoints
        log_dir: str = "./logs",                        # Directory to save training logs
        device: str = None                              # Device for training (auto-detect if None)
    ):
        # Store training configuration
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Move model to training device
        self.model.to(self.device)
        
        # Setup optimizer based on training mode
        self._setup_optimizer()
        
        # Create directories for saving models and logs
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging system
        self._setup_logging()
        
        # Training history for monitoring progress
        self.training_history = {
            'train_losses': [],      # Training loss per epoch
            'val_losses': [],        # Validation loss per epoch
            'learning_rates': [],    # Learning rate per epoch
            'epoch_times': []        # Training time per epoch
        }
        
        # Best model tracking for early stopping
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def _setup_optimizer(self):
        """Setup optimizer and determine which parameters to train based on model mode"""
        # Get parameters to optimize based on training mode
        if self.model.training_mode == 0:
            # Mode 0: Only optimize the MLP head parameters (base model frozen)
            params_to_optimize = list(self.model.traffic_head.parameters())
            print(f"Training mode 0: Optimizing {sum(p.numel() for p in params_to_optimize)} parameters (MLP head only)")
        else:
            # Mode 1: Optimize all model parameters (full fine-tuning)
            params_to_optimize = list(self.model.parameters())
            print(f"Training mode 1: Optimizing {sum(p.numel() for p in params_to_optimize)} parameters (full model)")
        
        # Filter out parameters that don't require gradients
        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]
        
        if not params_to_optimize:
            raise ValueError("No parameters to optimize! Check model configuration.")
        
        # Create AdamW optimizer with weight decay
        self.optimizer = AdamW(
            params_to_optimize,                 # Parameters to optimize
            lr=self.learning_rate,              # Learning rate
            weight_decay=self.weight_decay,     # Weight decay for regularization
            betas=(0.9, 0.999),                 # Standard Adam beta values
            eps=1e-8                            # Numerical stability epsilon
        )
    
    def _setup_logging(self):
        """Setup logging configuration for training monitoring"""
        log_file = os.path.join(self.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),    # Log to file
                logging.StreamHandler()           # Log to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training started. Logs will be saved to {log_file}")
    
    def train(
        self,
        num_epochs: int = 10,             # Total number of training epochs
        save_every: int = 5,              # Save checkpoint every N epochs
        eval_every: int = 1,              # Evaluate on validation set every N epochs
        early_stopping_patience: int = 5, # Stop training if no improvement for N epochs
        scheduler_type: str = "cosine"    # Learning rate scheduler: "cosine" or "plateau"
    ):
        """
        Train the network traffic model with comprehensive monitoring and checkpointing
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save model checkpoint every N epochs
            eval_every: Evaluate on validation set every N epochs
            early_stopping_patience: Stop training if no improvement for N epochs
            scheduler_type: Type of learning rate scheduler ("cosine" or "plateau")
            
        Returns:
            Dictionary containing training history (losses, learning rates, etc.)
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model training mode: {self.model.training_mode}")
        self.logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            self.logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        # Setup learning rate scheduler
        if scheduler_type == "cosine":
            # Cosine annealing: gradually reduce LR following cosine curve
            scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs * len(self.train_dataloader),
                eta_min=1e-7
            )
        else:
            # Plateau reduction: reduce LR when validation loss plateaus
            scheduler = ReduceLROnPlateau(
                self.optimizer,         # Optimizer to apply learning rate changes
                mode='min',             # Reduce LR when monitored metric stops decreasing
                factor=0.5,             # Reduce LR by a factor of 0.5
                patience=2,             # Wait 2 epochs with no improvement before reducing LR
                verbose=True            # Print a message when LR is reduced
            )
        
        # Early stopping counter
        no_improvement_count = 0
        
        # Main training loop
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            # Training phase
            train_loss = self._train_epoch(epoch, scheduler if scheduler_type == "cosine" else None)
            
            # Validation phase (if validation data provided)
            val_loss = None
            if self.val_dataloader and (epoch + 1) % eval_every == 0:
                val_loss = self._validate_epoch(epoch)
                
                # Update plateau scheduler based on validation loss
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                
                # Check for best model and update early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_path = self._save_checkpoint(epoch, val_loss, is_best=True)
                    no_improvement_count = 0
                    self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
                else:
                    no_improvement_count += 1
            
            # Save regular checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_loss)
            
            # Record training metrics for analysis
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_times'].append(epoch_time)
            
            # Log epoch summary
            log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            log_msg += f" - LR: {self.optimizer.param_groups[0]['lr']:.2e} - Time: {epoch_time:.1f}s"
            self.logger.info(log_msg)
            
            # Early stopping check
            if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {no_improvement_count} epochs without improvement")
                break
        
        # Save final model after training completion
        final_model_path = self._save_checkpoint(num_epochs - 1, val_loss, is_final=True)
        
        # Save training history for later analysis
        self._save_training_history()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best model: {self.best_model_path}")
        self.logger.info(f"Final model: {final_model_path}")
        
        return self.training_history
    
    def _train_epoch(self, epoch: int, scheduler=None) -> float:
        """Train for one epoch and return average training loss"""
        self.model.train()  # Set model to training mode
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        # Progress bar for training batches
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch data to training device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass through the model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass and optimization
            self.optimizer.zero_grad()    # Clear previous gradients
            loss.backward()               # Compute gradients
            
            # Apply gradient clipping to prevent exploding gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()         # Update model parameters
            
            # Update learning rate scheduler (for cosine annealing)
            if scheduler:
                scheduler.step()
            
            # Update progress bar with current metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch and return average validation loss"""
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        # Progress bar for validation batches
        progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Epoch {epoch+1} [Val]",
            leave=False
        )
        
        with torch.no_grad():  # Disable gradient computation for validation
            for batch in progress_bar:
                # Move batch data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (no gradient computation)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Update progress bar with current validation loss
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, val_loss: Optional[float], is_best: bool = False, is_final: bool = False) -> str:
        """Save model checkpoint with training state and metadata"""
        # Determine filename based on checkpoint type
        if is_best:
            filename = "best_model.pt"
        elif is_final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch+1}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Create comprehensive checkpoint with all training state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.training_history['train_losses'][-1] if self.training_history['train_losses'] else None,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_history': self.training_history,
            'model_config': {
                'model_name': self.model.model_name,
                'training_mode': self.model.training_mode,
                'n_features': self.model.n_features,
                'hidden_size': self.model.hidden_size
            }
        }
        
        torch.save(checkpoint, filepath)
        
        # Also save using model's save method for easy loading (best/final only)
        if is_best or is_final:
            model_path = filepath.replace('.pt', '_model.pt')
            self.model.save_model(model_path)
        
        return filepath
    
    def _save_training_history(self):
        """Save training history to JSON file for analysis and plotting"""
        history_path = os.path.join(self.log_dir, "training_history.json")
        
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, values in self.training_history.items():
            if values:
                # Handle None values and convert to float for JSON compatibility
                history_serializable[key] = [float(v) if v is not None else None for v in values]
            else:
                history_serializable[key] = []
        
        # Save training history as JSON
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from a previously saved checkpoint"""
        self.logger.info(f"Resuming training from {checkpoint_path}")
        
        # Load checkpoint from disk
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Restore best validation loss for early stopping
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            self.best_val_loss = checkpoint['val_loss']
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch'] + 1}")
        
        return checkpoint['epoch'] + 1
