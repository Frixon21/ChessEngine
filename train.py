# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_network import ChessNet
import math
from tqdm import tqdm
from torch.utils.data import Dataset

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
        board_tensor, target_policy, outcome = self.samples[idx]
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
        target_policy = torch.tensor(target_policy, dtype=torch.float32)
        outcome = torch.tensor(outcome, dtype=torch.float32)
        return board_tensor, target_policy, outcome


def train_network(data_source, num_epochs=10, batch_size=32, learning_rate=0.001, resume_from_checkpoint=None):  

    dataset = MemoryDataset(data_source)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    model = ChessNet().to(device)
    if resume_from_checkpoint is not None:
        print(f"Loading model weights from {resume_from_checkpoint}")
        model.load_state_dict(torch.load(resume_from_checkpoint))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    total_batches = num_epochs * math.ceil(len(dataset) / batch_size)
    pbar = tqdm(total=total_batches, desc="Overall Training Progress")
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        for board_tensor, target_policy, target_value in dataloader:
            board_tensor = board_tensor.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            
            optimizer.zero_grad()
            policy_out, value_out = model(board_tensor)
            
            log_policy = torch.log_softmax(policy_out, dim=1)
            policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
            value_loss = mse_loss(value_out.squeeze(), target_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            pbar.update(1)
            
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    pbar.close()
    checkpoint_path = "trained_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete, model saved as {checkpoint_path}")
    return checkpoint_path
