# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_network import ChessNet
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
# from torch.cuda.amp import autocast, GradScaler

class MemoryDataset(Dataset):
    def __init__(self, samples):
        """
        samples is a list of tuples: (board_tensor, target_policy, outcome)
        where each element is a numpy array or similar.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, policy, outcome = self.samples[idx]
        return torch.tensor(board, dtype=torch.float32), \
               torch.tensor(policy, dtype=torch.float32), \
               torch.tensor(outcome, dtype=torch.float32)



def train_network(data_source, num_epochs=10, batch_size=32, learning_rate=0.001, resume_from_checkpoint=None):  

    dataset = MemoryDataset(data_source)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")    
     
    model = ChessNet().to(device)    
    if resume_from_checkpoint is not None:
        print(f"Loading model weights from {resume_from_checkpoint}")
        model.load_state_dict(torch.load(resume_from_checkpoint, map_location=device))
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Example with weight decay
    
    hypothetical_total_epochs = 5
    total_steps = len(dataloader) * hypothetical_total_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6) # eta_min prevents LR going to absolute zero

    mse_loss = nn.MSELoss()
    
    # <<< AMP: Initialize GradScaler >>>
    # Creates gradient scaler for mixed precision. Enabled only if device is cuda.
    scaler = GradScaler(enabled=(device.type == 'cuda'))
        
    total_batches_per_epoch = len(dataloader)
    print(f"Starting training: {num_epochs} epochs, {total_batches_per_epoch} batches per epoch.")

    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_policy_loss = 0.0
        running_value_loss = 0.0 
        pbar = tqdm(enumerate(dataloader), total=total_batches_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (board_tensor, target_policy, target_value) in pbar:
            
            # if i == 0 and epoch == 0: # Print only once at the very start
            #     print("\n--- Sample Batch Data ---")
            #     print("Board Tensor Shape:", board_tensor.shape)
            #     print("Target Policy Shape:", target_policy.shape)
            #     print("Target Value Shape:", target_value.shape)
            #     # Check policy sums (should be close to 1)
            #     print("Target Policy Sum (Sample 0):", torch.sum(target_policy[0]).item())
            #     # Check value range (should be [-1, 1])
            #     print("Target Value Range (Batch):", torch.min(target_value).item(), torch.max(target_value).item())
            #     # Check for NaNs/Infs
            #     print("NaNs in Board:", torch.isnan(board_tensor).any().item())
            #     print("NaNs in Policy Target:", torch.isnan(target_policy).any().item())
            #     print("NaNs in Value Target:", torch.isnan(target_value).any().item())
            #     print("--- End Sample Batch Data ---")           
            
            # Move data to device (non_blocking=True useful with pin_memory)
            board_tensor = board_tensor.to(device, non_blocking=True)
            target_policy = target_policy.to(device, non_blocking=True)
            target_value = target_value.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing
            
            # <<< AMP: Enable autocast context manager >>>
            # Runs the forward pass under mixed precision
            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                policy_out, value_out = model(board_tensor)
                log_policy = torch.log_softmax(policy_out, dim=1)
                policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
                value_loss = mse_loss(value_out.squeeze(-1), target_value)
                loss = policy_loss + value_loss
                
            # Check for NaN loss immediately after calculation
            if torch.isnan(loss):
                print(f"\n!!! NaN loss detected at Epoch {epoch+1}, Batch {i}. Skipping batch. !!!")
                print("Inputs causing NaN:", board_tensor)
                print("Targets causing NaN:", target_policy, target_value)
                print("Outputs causing NaN:", policy_out, value_out)
                # Need to reset optimizer state potentially, or just skip update
                optimizer.zero_grad() # Re-zero just in case
                continue # Skip the rest of the loop for this batch
                
            # if i == 0 and epoch % 5 == 0: # Print every 5 epochs for first batch
            #     print("\n--- Sample Model Output vs Target (Batch 0) ---")
            #     # Compare value prediction to target for first few samples
            #     print("Value Out vs Target:")
            #     # Ensure value_out exists and has data before accessing
            #     if 'value_out' in locals() and value_out.numel() > 0:
            #          for k in range(min(5, value_out.shape[0])):
            #              print(f"  Pred: {value_out[k].item():.4f}, Target: {target_value[k].item():.4f}")
            #     else: print("  (value_out not available)")

            #     # Compare top predicted policy move vs target policy for first sample
            #     print("\nPolicy Out vs Target (Sample 0):")
            #     if 'policy_out' in locals() and policy_out.numel() > 0:
            #          try:
            #               policy_probs_out = torch.softmax(policy_out[0], dim=0)
            #               top_pred_idx = torch.argmax(policy_probs_out).item()
            #               top_pred_prob = policy_probs_out[top_pred_idx].item()
            #               top_target_idx = torch.argmax(target_policy[0]).item()
            #               top_target_prob = target_policy[0][top_target_idx].item()
            #               # You'd need move_index_to_move function here for readability
            #               print(f"  Top Pred Idx: {top_pred_idx} (Prob: {top_pred_prob:.4f})")
            #               print(f"  Top Target Idx: {top_target_idx} (Prob: {top_target_prob:.4f})")
            #          except Exception as e: print(f"  Error printing policy details: {e}")
            #     else: print("  (policy_out not available)")
            #     print("--- End Sample Model Output ---")
                
                
            # <<< AMP: Scale loss and call backward >>>
            # scaler.scale multiplies the loss by the scale factor
            scaler.scale(loss).backward()
            
            
            # # Optional gradient check (can slow down training)
            # scaler.unscale_(optimizer) # Need to unscale first if using AMP
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0) # Clip and get norm
            # print(f"Grad Norm: {grad_norm.item()}")
            # # Or check a specific layer:
            # policy_grad_mean = model.policy_fc.weight.grad.abs().mean().item()
            # print(f"Policy FC Grad Mean: {policy_grad_mean}")
            # optimizer.zero_grad() # Zero grads *after* unscaling/checking if you do this manually
            
            # Optional: Gradient Clipping (uncomment if gradients explode)
            # scaler.unscale_(optimizer) # Unscales gradients before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # <<< AMP: Scaler step and update >>>
            # scaler.step examines gradients and updates weights if no NaNs/Infs
            scaler.step(optimizer)
            # scaler.update updates the scale factor for the next iteration
            scaler.update()
            
            # <<< Scheduler: Step per BATCH (after optimizer step) >>>
            scheduler.step()
                        
            running_loss += loss.item()
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()   
            pbar.set_postfix(loss=running_loss / (i + 1))

            
        avg_total_loss = running_loss / total_batches_per_epoch
        avg_policy_loss = running_policy_loss / total_batches_per_epoch
        avg_value_loss = running_value_loss / total_batches_per_epoch
        current_lr = scheduler.get_last_lr()[0] # Get current learning rate
        print(f"Epoch {epoch+1}/{num_epochs} finished. Avg Loss: {avg_total_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}), LR: {current_lr:.6f}")

    
    checkpoint_path = "trained_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete, model saved as {checkpoint_path}")
    return checkpoint_path
