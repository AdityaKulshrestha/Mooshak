import os
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler
from model import Llama
from utils import LoadTextCorpus
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.training.distributed:
        world_size = cfg.training.num_devices
        mp.spawn(
            train_fsdp,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
    else:
        train_fsdp(0, 1, cfg)

def setup_fsdp(rank: int, world_size: int, cfg: DictConfig) -> None:
    """Initialize distributed training environment for FSDP"""
    os.environ['MASTER_ADDR'] = cfg.distributed.master_addr
    os.environ['MASTER_PORT'] = str(cfg.distributed.master_port)
    os.environ['PT_HPU_LAZY_MODE'] = '0'
    
    import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def create_fsdp_model(cfg: DictConfig, device: torch.device) -> FSDP:
    """Create and wrap model with FSDP"""
    model = Llama(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        intermediate_dim=cfg.model.intermediate_dim
    ).to(device)

    return FSDP(
        model,
        device_id=device,
        **cfg.fsdp
    )

def train_fsdp(rank: int, world_size: int, cfg: DictConfig) -> None:
    """Main FSDP training process"""
    setup_fsdp(rank, world_size, cfg)
    device = torch.device(cfg.training.device)
    
    # Model setup
    model = create_fsdp_model(cfg, device)
    optimizer = FusedAdamW(
        model.parameters(),
        lr=cfg.optimizer.base_lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay
    )
    scheduler = hydra.utils.instantiate(cfg.optimizer.scheduler, optimizer=optimizer)
    
    # Data loading
    train_loader, val_loader = create_fsdp_data_loaders(rank, world_size, cfg)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg.training.epochs):
        train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Training step
            loss = train_step(model, optimizer, scheduler, x, y, cfg)
            
            # Logging and validation
            if rank == 0 and batch_idx % cfg.logging.interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.training.epochs} | "
                    f"Batch {batch_idx} | Loss: {loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            
            if should_validate(batch_idx, cfg):
                val_loss = validate_fsdp(model, val_loader, device, cfg)
                handle_validation_results(rank, epoch, batch_idx, val_loss, best_val_loss, model, optimizer, cfg)
            
            if should_save_checkpoint(batch_idx, cfg):
                save_fsdp_checkpoint(rank, model, optimizer, epoch, batch_idx, loss, cfg)
    
    cleanup_distributed()

def create_fsdp_data_loaders(rank: int, world_size: int, cfg: DictConfig):
    """Create distributed data loaders for FSDP"""
    train_data = LoadTextCorpus(cfg.data.train_path, cfg.model.block_size)
    val_data = LoadTextCorpus(cfg.data.val_path, cfg.model.block_size)
    
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
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

def train_step(model: FSDP, optimizer: FusedAdamW, scheduler, x: torch.Tensor, y: torch.Tensor, cfg: DictConfig):
    """Perform a single FSDP training step"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    # Backward pass
    loss.backward()
    htcore.mark_step()
    
    # Optimization
    optimizer.step()
    scheduler.step()
    htcore.mark_step()
    
    return loss.item()

def validate_fsdp(model: FSDP, val_loader: DataLoader, device: torch.device, cfg: DictConfig):
    """FSDP validation process"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            htcore.mark_step()
    
    # Synchronize across processes
    dist.all_reduce(torch.tensor(total_loss, device=device))
    dist.all_reduce(torch.tensor(total_samples, device=device))
    
    return total_loss / total_samples

def save_fsdp_checkpoint(rank: int, model: FSDP, optimizer, epoch: int, batch: int, loss: float, cfg: DictConfig):
    """Save FSDP checkpoint with proper shard handling"""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    checkpoint_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        cfg.checkpoint.dir
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = f"fsdp_checkpoint_epoch{epoch}_batch{batch}.pt"
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    logger.info(f"Saved FSDP checkpoint to {filename}")

def cleanup_distributed():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()


if __name__ == "__main__":
    main()