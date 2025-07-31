"""
Network Traffic Model Training Script

WHAT THIS SCRIPT DOES:
This script trains an AI model to understand and generate network traffic patterns.
Think of it like teaching a computer to recognize different types of internet traffic
(like YouTube videos, Facebook posts, Google searches, etc.) and then create similar
artificial traffic data for testing purposes.

TWO WAYS TO TRAIN:
- Mode 0: Quick Training (Frozen base + MLP head)
  * Faster to train (takes less time)
  * Uses less computer memory
  * Good for quick experiments
  
- Mode 1: Deep Training (Full fine-tuning)
  * Slower to train (takes more time)
  * Uses more computer memory
  * Usually gives better results

HOW TO USE:
1. Put your network traffic data in CSV format in the 'dataset' folder
2. Update the DATA_PATH below to point to your file
3. Adjust the training settings if needed (or keep defaults)
4. Run: python training_stage.py
5. Check the results in 'trained_models' and 'training_logs' folders

WHAT YOU GET:
- A trained AI model that understands your network traffic
- Training plots showing how well the model learned
- Log files with detailed training information
- Model checkpoints you can use later
"""

import os
import sys
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network_traffic_generator.model import NetworkTrafficModel
from network_traffic_generator.dataset import (
    load_network_traffic_data, 
    split_data, 
    create_data_loader,
    analyze_data_distribution
)
from network_traffic_generator.trainer import NetworkTrafficTrainer

# =============================================================================
# TRAINING SETTINGS - CHANGE THESE TO CUSTOMIZE YOUR TRAINING
# =============================================================================

# DATA SETTINGS
DATA_PATH = "dataset/my_dataset.csv"     #Put your dataset here

# MODEL SETTINGS
MODEL_NAME = "gpt2"     # Options: "gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"
TRAINING_MODE = 1       # 0 = Freeze mode; 1 = Full fine-tuning 
HIDDEN_SIZE = 768       # Internal model size

# TRAINING BEHAVIOR
NUM_EPOCHS = 15         # How many times to go through all your data
BATCH_SIZE = 4          # How many examples to process at once
LEARNING_RATE = 3e-5    # How fast the model learns 
WEIGHT_DECAY = 0.01     # Prevents the model from becoming too specialized
MAX_LENGTH = 512        # Maximum length of data sequences to process
WARMUP_STEPS = 100      # Gradual learning rate increase at the start

# DATA SPLITTING
TRAIN_RATIO = 0.85      # 85% of data used for training
VAL_RATIO = 0.15        # 15% of data used for validation

# Directories
SAVE_DIR = "./trained_model"
LOG_DIR = "./training_logs"

# Training options
SAVE_EVERY = 5                  # Save checkpoint every N epochs
EVAL_EVERY = 1                  # Evaluate on validation set every N epochs
EARLY_STOPPING_PATIENCE = 5     # Stop if no improvement for N epochs
SCHEDULER_TYPE = "plateau"      # "cosine" or "plateau"

# Data augmentation
AUGMENT_DATA = False            # Apply data augmentation during training

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """
    This is the main function that does all the training work.
    It follows these steps:
    1. Load and check your data
    2. Split data into training and validation sets
    3. Set up the AI model
    4. Create data loaders (tools to feed data to the model)
    5. Set up the trainer (the thing that teaches the model)
    6. Actually train the model
    7. Create plots to show how training went
    8. Save everything for later use
    """
    
    print("=" * 60)
    print("NETWORK TRAFFIC MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Training Mode: {TRAINING_MODE} ({'Frozen base + MLP' if TRAINING_MODE == 0 else 'Full fine-tuning'})")
    print(f"LLM Model: {MODEL_NAME}")
    print(f"Data Source: {DATA_PATH}")
    print()
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # Create folders to save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Start the timer to see how long training takes
    start_time = time.time()

    try:
        # STEP 1: Load and examine the data
        print("Step 1: Loading and examining your data...")
        data = load_network_traffic_data(DATA_PATH)
        
        # Analyze data distribution
        data_analysis = analyze_data_distribution(data)
        print(f"Total network traffic samples: {data_analysis['overall']['total_samples']}")
        print(f"Number of features per sample: {data_analysis['overall']['total_features']}")
        print(f"Different types of apps found: {data_analysis['app_labels']['unique_count']}")
        print(f"Most common app in data: {data_analysis['app_labels']['most_common']}")
        print()
        
        # STEP 2: Split the data for training and testing
        print("Step 2: Splitting data for training and validation...")
        train_data, val_data, test_data = split_data(
            data, 
            train_ratio=TRAIN_RATIO,    # Most data for training
            val_ratio=VAL_RATIO,        # Some data to check progress
            test_ratio=0.0,             # No separate test set needed
            random_state=42             # Random seed for consistent results
        )
        print(f"Data split complete:")
        print(f"  - Training samples: {len(train_data)} ({TRAIN_RATIO*100:.0f}%)")
        print(f"  - Validation samples: {len(val_data)} ({VAL_RATIO*100:.0f}%)")
        print()
        
        # STEP 3: Set up the AI model
        print("Step 3: Setting up the AI model...")
        model = NetworkTrafficModel(
            model_name=MODEL_NAME,          # Which pre-trained model to use
            training_mode=TRAINING_MODE,    # How to train it (quick vs deep)
            n_features=91,                  # Number of features in your data (90 network + 1 app)
            hidden_size=HIDDEN_SIZE         # Internal model complexity
        )
        
        # Show information about the model 
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model information:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Parameters that will be trained: {trainable_params:,}")
        print(f"  - Training ratio: {trainable_params/total_params:.2%}")
        print()
        
        # STEP 4: Set up data loaders
        print("Step 4: Setting up data loaders...")
        train_loader = create_data_loader(
            train_data,                     # The training data
            model=model,                    # The model that will use this data
            batch_size=BATCH_SIZE,          # How many samples to process at once
            shuffle=True,                   # Mix up the order (good for learning)
            max_length=MAX_LENGTH,          # Maximum sequence length
            augment_data=AUGMENT_DATA,      # Add artificial variations if enabled
            num_workers=1                   # Number of parallel processes
        )
        
        val_loader = create_data_loader(
            val_data,                       # The validation data
            model=model,                    # The model that will use this data
            batch_size=BATCH_SIZE,          # How many samples to process at once
            shuffle=False,                  # Don't mix up validation data
            max_length=MAX_LENGTH,          # Maximum sequence length
            augment_data=False,             # No artificial variations for validation
            num_workers=1                   # Number of parallel processes
        )
        print("Data loaders ready!")
        print()
        
        # STEP 5: Set up the trainer
        print("Step 5: Setting up the trainer...")
        trainer = NetworkTrafficTrainer(
            model=model,                    # The model to train
            train_dataloader=train_loader,  # Training data feeder
            val_dataloader=val_loader,      # Validation data feeder
            learning_rate=LEARNING_RATE,    # How fast to learn
            weight_decay=WEIGHT_DECAY,      # Prevents overfitting
            warmup_steps=WARMUP_STEPS,      # Gradual learning rate increase
            save_dir=SAVE_DIR,              # Where to save trained models
            log_dir=LOG_DIR,                # Where to save training logs
            device=DEVICE                   # CPU or GPU
        )
        print("Trainer ready!")
        print()
        
        # STEP 6: Train the model!
        print("Step 6: Starting the training process...")
        print("This might take a while, so grab a coffee :)")
        print("-" * 60)
        
        training_history = trainer.train(
            num_epochs=NUM_EPOCHS,                              # How many complete passes through the data
            save_every=SAVE_EVERY,                              # Save progress every N epochs
            eval_every=EVAL_EVERY,                              # Check validation performance every N epochs
            early_stopping_patience=EARLY_STOPPING_PATIENCE,    # Stop if no improvement for N epochs
            scheduler_type=SCHEDULER_TYPE                       # How to adjust learning rate
        )
        
        print("-" * 60)
        print("Training completed successfully!")
        
        # Step 7: Generate training plots
        print("\nStep 7: Generating training analysis plots...")
        model_info = {
            'model_name': MODEL_NAME,
            'training_mode': f"Mode {TRAINING_MODE} ({'Frozen base + MLP' if TRAINING_MODE == 0 else 'Full fine-tuning'})",
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params/total_params,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
        
        plot_path, convergence_path = create_training_plots(training_history, SAVE_DIR, model_info)
        
        # STEP 8: Show final results and summary
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        print(f"✓ Best validation loss achieved: {trainer.best_val_loss:.4f}")
        print(f"✓ Best model saved at: {trainer.best_model_path}")
        print(f"✓ Final training loss: {training_history['train_losses'][-1]:.4f}")
        if training_history['val_losses'][-1] is not None:
            print(f"✓ Final validation loss: {training_history['val_losses'][-1]:.4f}")
        
        # Show how much the model improved during training
        if len(training_history['train_losses']) > 1:
            initial_loss = training_history['train_losses'][0]
            final_loss = training_history['train_losses'][-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"Overall training improvement: {improvement:.1f}%")
            
            # Check if the model is still getting better or has stabilized
            recent_losses = training_history['train_losses'][-3:]
            if len(recent_losses) >= 3:
                trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                if abs(trend) < 0.001:
                    print("Model appears to have converged (reached stability)")
                elif trend < 0:
                    print("Model was still improving when training ended")
                else:
                    print("Model may be having difficulty (loss was increasing)")
        
        print(f"\nFiles created:")
        print(f"   * Detailed analysis plots: {plot_path}")
        print(f"   * Simple convergence plot: {convergence_path}")
        
        # Save a text file with all the training details
        config_path = os.path.join(SAVE_DIR, "training_config.txt")
        elapsed = time.time() - start_time

        # Convert elapsed time to human-readable format
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_str = f"{hours}h {minutes}m {seconds}s"

        with open(config_path, 'w') as f:
            f.write("NETWORK TRAFFIC MODEL TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write("WHEN AND WHERE:\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {DATA_PATH}\n")
            f.write(f"Results saved in: {SAVE_DIR}\n\n")
            f.write("MODEL DETAILS:\n")
            f.write(f"AI model used: {MODEL_NAME}\n")
            f.write(f"Training mode: {TRAINING_MODE} ({'Quick Mode' if TRAINING_MODE == 0 else 'Deep Mode'})\n")
            f.write(f"Total model parameters: {total_params:,}\n")
            f.write(f"Parameters that were trained: {trainable_params:,}\n\n")
            f.write("TRAINING SETTINGS:\n")
            f.write(f"Number of epochs: {NUM_EPOCHS}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Weight decay: {WEIGHT_DECAY}\n\n")
            f.write("RESULTS:\n")
            f.write(f"Best validation loss: {trainer.best_val_loss:.4f}\n")
            f.write(f"Total training time: {elapsed_str}\n")

        print(f"Configuration saved to: {config_path}")
        print(f"\nTotal training time: {elapsed_str}")

        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

# =============================================================================
# VISUALIZATION FUNCTIONS - CREATES CHARTS TO SHOW TRAINING PROGRESS
# =============================================================================

def create_training_plots(training_history, save_dir, model_info):
    """
    Creates visual charts showing how well the training went.
    
    WHAT THIS CREATES:
    - Loss curves: Shows how the model's errors decreased over time
    - Convergence plots: Shows if the model reached stability
    - Learning rate charts: Shows how the learning speed changed
    - Performance summaries: Text summary of results
    """

    print("Creating visual training reports...")
    
    # Set up the plotting style for nice-looking charts
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 100
    
    # Get the training data to plot
    epochs = range(1, len(training_history['train_losses']) + 1)
    train_losses = training_history['train_losses']
    val_losses = [loss for loss in training_history['val_losses'] if loss is not None]
    val_epochs = [i+1 for i, loss in enumerate(training_history['val_losses']) if loss is not None]
    
    # Create a large figure with multiple charts
    fig = plt.figure(figsize=(16, 12))
    
    # CHART 1: Basic loss curves (most important chart)
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CHART 2: Smoothed loss curves 
    plt.subplot(2, 3, 2)
    if len(train_losses) > 5:
        # Apply smoothing to reduce noise and show clear trends
        window = min(5, len(train_losses) // 3)
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window-1:]
        plt.plot(smooth_epochs, train_smooth, 'b-', label=f'Training (MA-{window})', linewidth=2)
    else:
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses and len(val_losses) > 3:
        window_val = min(3, len(val_losses) // 2)
        if len(val_losses) >= window_val:
            val_smooth = np.convolve(val_losses, np.ones(window_val)/window_val, mode='valid')
            val_smooth_epochs = val_epochs[window_val-1:]
            plt.plot(val_smooth_epochs, val_smooth, 'r-', label=f'Validation (MA-{window_val})', linewidth=2)
    elif val_losses:
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Smoothed)')
    plt.title('Loss Convergence (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CHART 3: Learning rate changes (if available)
    plt.subplot(2, 3, 3)
    if 'learning_rates' in training_history and training_history['learning_rates']:
        plt.plot(epochs, training_history['learning_rates'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('How Fast the Model Learned')
        plt.yscale('log')  # Use log scale because learning rates are very small
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Learning Rate Schedule')
    
    # CHART 4: Distribution of loss values
    plt.subplot(2, 3, 4)
    plt.hist(train_losses, bins=20, alpha=0.7, label='Training', color='blue', density=True)
    if val_losses:
        plt.hist(val_losses, bins=15, alpha=0.7, label='Validation', color='red', density=True)
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CHART 5: Improvement over time (percentage improvement from start)
    plt.subplot(2, 3, 5)
    if len(train_losses) > 1:
        train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
        plt.plot(epochs, train_improvement, 'b-', label='Training Improvement', linewidth=2)
    
    if len(val_losses) > 1:
        val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        plt.plot(val_epochs, val_improvement, 'r-', label='Validation Improvement', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Improvement (%)')
    plt.title('Loss Improvement from Start')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # CHART 6: Text summary of training results
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create a detailed text summary
    info_text = f"""Training Summary Report
    
    Model: {model_info.get('model_name', 'N/A')}
    Training Mode: {model_info.get('training_mode', 'N/A')}
    Total Parameters: {model_info.get('total_params', 'N/A'):,}
    Trainable Parameters: {model_info.get('trainable_params', 'N/A'):,}
    Trainable Ratio: {model_info.get('trainable_ratio', 'N/A'):.1%}

    Training Configuration:
    Epochs: {len(train_losses)}
    Batch Size: {model_info.get('batch_size', 'N/A')}
    Learning Rate: {model_info.get('learning_rate', 'N/A')}
    Weight Decay: {model_info.get('weight_decay', 'N/A')}

    Results:
    Final Train Loss: {train_losses[-1]:.4f}
    Best Train Loss: {min(train_losses):.4f}
    """
    
    if val_losses:
        final_loss = val_losses[-1]
        best_loss = min(val_losses)
        info_text += f"Final Val Loss: {final_loss:.4f}\nBest Val Loss: {best_loss:.4f}\n"
    
    # Add a warning if the model might be overfitting
    if len(val_losses) > 3 and len(train_losses) > 3:
        recent_train = np.mean(train_losses[-3:])
        recent_val = np.mean(val_losses[-3:])
        if recent_val > recent_train * 1.2:
            info_text += "\nPossible Overfitting Detected"
        elif recent_val < recent_train * 1.1:
            info_text += "\nGood Generalization"
    
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f"training_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")
    
    # Create a simple convergence plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Model Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add helpful annotations to show key points
    if len(train_losses) > 1:
        final_loss = train_losses[-1]
        best_loss = min(train_losses)
        improvement = (train_losses[0] - final_loss) / train_losses[0] * 100
        
        # Annotate the final performance
        ax.annotate(f'Final: {final_loss:.4f}', 
                   xy=(len(train_losses), final_loss), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7, color="white"))
        
        # Annotate the best performance
        ax.annotate(f'Best: {best_loss:.4f}', 
                   xy=(train_losses.index(best_loss) + 1, best_loss), xytext=(10, -20),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7, color="white"))
    
    convergence_path = os.path.join(save_dir, f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {convergence_path}")
    
    return plot_path, convergence_path

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
