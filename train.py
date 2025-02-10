import os
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import DistAwareColoringTransformer , multiple_inference_passes_parallel
from loss import coloring_loss
from data_utils import load_dimacs_graphs


def train_attention_with_scheduler(
    folder="3-col",
    num_colors=3,
    embed_dim=128,
    n_layers=10,
    lr=1e-3,
    epochs=5,
    device='cpu',
    step_size=2,
    gamma=0.5,
    save_dir="checkpoints",
    save_filename="best_model.pth",
    passes=10  # Number of inference passes during validation
):
    """
    Demonstration of training with a StepLR scheduler and model checkpointing.
      - step_size: how many epochs between each lr step
      - gamma: multiplicative factor of LR decay
      - save_dir: directory to save model checkpoints
      - save_filename: filename for the best model checkpoint
      - passes: number of inference passes during validation
    """
    # 1) Load dataset
    dataset = load_dimacs_graphs(folder)  # a list of (num_nodes, edges)
    random.shuffle(dataset)
    #split_idx = int(0.8 * 5000)
    train_data = dataset[:4000]
    val_data   = dataset[4000:5000]
    print(f"Train set: {len(train_data)} graphs, Validation set: {len(val_data)} graphs.")

    # 2) Build model & optimizer
    model = DistAwareColoringTransformer(
        embed_dim=embed_dim,
        num_colors=num_colors,
        n_layers=n_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) Create the scheduler
    # Here, we use StepLR with (step_size, gamma) as hyperparams
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 4) Prepare for checkpointing
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    best_avg_unsat = float('inf')
    best_epoch = -1

    # Optionally, save initial model state
    # torch.save(model.state_dict(), os.path.join(save_dir, "initial_model.pth"))

    # 5) Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (n_nodes, edges) in train_data:
            if n_nodes < 1:
                continue

            # Forward pass
            color_probs = model(n_nodes, edges)
            loss = coloring_loss(color_probs, edges)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_data))

        # -- Validation (with multiple inference passes) --
        model.eval()
        total_val_loss = 0.0
        total_unsat = 0.0
        with torch.no_grad():
            for (n_nodes, edges) in val_data:
                if n_nodes < 1:
                    continue

                # Run multiple inference passes to find the best coloring
                best_probs, best_unsat_pct = multiple_inference_passes_parallel(
                    model, n_nodes, edges, passes=passes
                )
                val_loss = coloring_loss(best_probs, edges)
                total_val_loss += val_loss.item()
                total_unsat += best_unsat_pct

        avg_val_loss = total_val_loss / max(1, len(val_data))
        avg_val_unsat = total_unsat / max(1, len(val_data))

        # Check if current epoch has the best validation unsatisfied percentage
        if avg_val_unsat < best_avg_unsat:
            best_avg_unsat = avg_val_unsat
            best_epoch = epoch + 1
            # Save the best model
            save_path = os.path.join(save_dir, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f"--> New best model found at Epoch {best_epoch} with Avg Val Unsat: {best_avg_unsat:.2f}%. Model saved to {save_path}")

        # Step the scheduler at the end of each epoch
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]  # If only one param group
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Unsat: {avg_val_unsat:.2f}%")

    print(f"\nTraining completed. Best Avg Val Unsat: {best_avg_unsat:.2f}% at Epoch {best_epoch}.")

    # Optionally, load the best model before returning
    # model.load_state_dict(torch.load(os.path.join(save_dir, save_filename)))
    # return model

    return model
