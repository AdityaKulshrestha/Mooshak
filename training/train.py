import os
import math
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from model import Llama
from utils import LoadTextCorpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self):
        self.train_data = 'data/train.bin'
        self.val_data = 'data/val.bin'
        self.epochs = 10
        self.save_dir = 'checkpoints'
        self.device = torch.device('hpu')
        self.batch_size = 8
        self.block_size = 2048
        self.min_lr = 3e-5
        self.max_lr = 3e-4
        self.save_interval = 1000
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.vocab_size = 64128
        self.warmup_iters = 3000
        self.lr_decay_iters = 600000
        self.num_workers = 8

def get_learning_rate(iter_num: int, config: TrainingConfig) -> float:
    """Cosine learning rate decay with warmup"""
    if iter_num < config.warmup_iters:
        return config.max_lr * iter_num / config.warmup_iters
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)

def validate_model(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Run model validation"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            htcore.mark_step()
    
    return total_loss / total_samples

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   iteration: int, loss: float, config: TrainingConfig):
    """Save training checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.save_dir,
        f"model_iter_{iteration}_loss_{loss:.4f}.pth"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def initialize_training(config: TrainingConfig):
    """Initialize model, optimizer, and data loaders"""
    # Model setup
    model = Llama(vocab_size=config.vocab_size, seq_len=config.block_size)
    model = model.to(config.device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    # Optimizer setup
    optimizer = FusedAdamW(
        model.parameters(),
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    # Data loading
    train_dataset = LoadTextCorpus(config.train_data, config.block_size)
    val_dataset = LoadTextCorpus(config.val_data, config.block_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True
    )

    return model, optimizer, train_loader, val_loader

def train_loop(config: TrainingConfig):
    """Main training loop"""
    model, optimizer, train_loader, val_loader = initialize_training(config)
    
    progress_bar = tqdm(total=len(train_loader)*config.epochs, desc="Training Progress")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            iteration = epoch * len(train_loader) + batch_idx
            
            # Learning rate scheduling
            lr = get_learning_rate(iteration, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Move data to device
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Backward pass
            loss.backward()
            htcore.mark_step()
            
            # Optimization
            optimizer.step()
            htcore.mark_step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({
                'epoch': f"{epoch+1}/{config.epochs}",
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr:.2e}"
            })
            progress_bar.update(1)
            
            # Validation and checkpointing
            if iteration % config.save_interval == 0 and iteration > 0:
                val_loss = validate_model(model, val_loader, config.device)
                save_checkpoint(model, optimizer, iteration, val_loss, config)
                
    progress_bar.close()

if __name__ == "__main__":
    config = TrainingConfig()   # Replace this with hydra
    train_loop(config)