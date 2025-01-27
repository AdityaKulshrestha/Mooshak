import os
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import Llama
from utils import LoadTextCorpus
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

# Configure logging
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.training.distributed:
        world_size = cfg.training.num_devices
        mp.spawn(
            train_process,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
    else:
        train_process(0, 1, cfg)

def setup_distributed(rank: int, world_size: int, cfg: DictConfig) -> None:
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = cfg.distributed.master_addr
    os.environ['MASTER_PORT'] = str(cfg.distributed.master_port)
    
    if cfg.distributed.backend == "hccl":
        import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(
        backend=cfg.distributed.backend,
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def create_model(cfg: DictConfig) -> torch.nn.Module:
    """Initialize model from config"""
    return Llama(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        intermediate_dim=cfg.model.intermediate_dim
    ).to(cfg.training.device)

def validate(model: torch.nn.Module, val_loader: DataLoader, cfg: DictConfig) -> float:
    """Run validation on the model"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(cfg.training.device), y.to(cfg.training.device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            htcore.mark_step()

    if cfg.training.distributed:
        dist.all_reduce(torch.tensor(total_loss, device=cfg.training.device))
        dist.all_reduce(torch.tensor(total_samples, device=cfg.training.device))
    
    return total_loss / total_samples

def train_process(rank: int, world_size: int, cfg: DictConfig) -> None:
    """Main training process"""
    if cfg.training.distributed:
        setup_distributed(rank, world_size, cfg)
    
    # Initialize model and optimizer
    model = create_model(cfg)
    optimizer = FusedAdamW(
        model.parameters(),
        lr=cfg.optimizer.base_lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay
    )
    
    scheduler = hydra.utils.instantiate(
        cfg.optimizer.scheduler,
        optimizer=optimizer
    )
    
    # Wrap model for distributed training
    if cfg.training.distributed:
        model = DDP(model, find_unused_parameters=True)
    
    # Load datasets
    train_data = LoadTextCorpus(cfg.data.train_path, cfg.model.block_size)
    val_data = LoadTextCorpus(cfg.data.val_path, cfg.model.block_size)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, rank, world_size, cfg
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg.training.epochs):
        if cfg.training.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            loss = train_step(model, optimizer, scheduler, x, y, cfg)
            
            # Logging and validation
            if rank == 0 and batch_idx % cfg.logging.interval == 0:
                log_progress(epoch, batch_idx, loss, scheduler, cfg)
            
            if should_validate(batch_idx, cfg):
                val_loss = validate(model.module, val_loader, cfg)
                handle_validation_results(rank, epoch, batch_idx, val_loss, best_val_loss, model, optimizer, cfg)
            
            if should_save_checkpoint(batch_idx, cfg):
                save_checkpoint(rank, epoch, batch_idx, loss, model, optimizer, cfg)
    
    if cfg.training.distributed:
        cleanup_distributed()

def train_step(model, optimizer, scheduler, x, y, cfg):
    """Perform a single training step"""
    model.train()
    x, y = x.to(cfg.training.device), y.to(cfg.training.device)
    
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
    
    # Backward pass
    loss.backward()
    htcore.mark_step()
    
    # Optimization
    optimizer.step()
    scheduler.step()
    htcore.mark_step()
    
    return loss.item()

def create_data_loaders(train_data, val_data, rank, world_size, cfg):
    """Create distributed data loaders"""
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank
    ) if cfg.training.distributed else None
    
    val_sampler = DistributedSampler(
        val_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if cfg.training.distributed else None
    
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.training.batch_size,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        drop_last=True
    )
    
    return train_loader, val_loader

def log_progress(epoch, batch_idx, loss, scheduler, cfg):
    """Log training progress"""
    logger.info(
        f"Epoch {epoch+1}/{cfg.training.epochs} | "
        f"Batch {batch_idx} | "
        f"Loss: {loss:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )

def should_validate(batch_idx, cfg):
    """Check if validation should be performed"""
    return batch_idx % cfg.validation.interval == 0 and batch_idx > 0

def handle_validation_results(rank, epoch, batch_idx, val_loss, best_val_loss, model, optimizer, cfg):
    """Handle validation results and model saving"""
    if rank == 0:
        logger.info(f"Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                rank,
                epoch,
                batch_idx,
                val_loss,
                model,
                optimizer,
                cfg,
                "best_model"
            )

def should_save_checkpoint(batch_idx, cfg):
    """Check if checkpoint should be saved"""
    return batch_idx % cfg.checkpoint.save_interval == 0

def save_checkpoint(rank, epoch, batch, loss, model, optimizer, cfg, suffix=None):
    """Save training checkpoint"""
    if rank != 0:
        return
        
    checkpoint_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        cfg.checkpoint.dir
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = f"checkpoint_epoch{epoch}_batch{batch}.pt" if not suffix else f"{suffix}.pt"
    path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.module.state_dict() if cfg.training.distributed else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, path)
    
    logger.info(f"Saved checkpoint to {path}")

def cleanup_distributed():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

if __name__ == "__main__":
    main()