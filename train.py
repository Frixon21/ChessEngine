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
    
    hypothetical_total_epochs = 10
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
        pbar = tqdm(enumerate(dataloader), total=total_batches_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (board_tensor, target_policy, target_value) in pbar:
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
                # Ensure value_out is squeezed correctly for MSELoss
                value_loss = mse_loss(value_out.squeeze(-1), target_value)
                loss = policy_loss + value_loss
                
            # <<< AMP: Scale loss and call backward >>>
            # scaler.scale multiplies the loss by the scale factor
            scaler.scale(loss).backward()
            
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
            pbar.set_postfix(loss=running_loss / (i + 1))

            
        avg_loss = running_loss / total_batches_per_epoch
        current_lr = scheduler.get_last_lr()[0] # Get current learning rate
        print(f"Epoch {epoch+1}/{num_epochs} finished. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    checkpoint_path = "trained_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete, model saved as {checkpoint_path}")
    return checkpoint_path
