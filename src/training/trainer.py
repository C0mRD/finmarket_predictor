# src/training/trainer.py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Dict
import wandb
from ..models.multimodal_transformer import MultiModalTransformer

class MarketPredictor(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model = MultiModalTransformer(config)
        
    def forward(self, batch):
        return self.model(
            batch['market_data'],
            batch['news_data'],
            batch['social_data']
        )
    
    def training_step(self, batch, batch_idx):
        mean, log_var = self(batch)
        loss = self.gaussian_nll_loss(
            batch['target'],
            mean,
            log_var.exp()
        )
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mean, log_var = self(batch)
        loss = self.gaussian_nll_loss(
            batch['target'],
            mean,
            log_var.exp()
        )
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs']
        )
        return [optimizer], [scheduler]
    
    @staticmethod
    def gaussian_nll_loss(target, mean, variance):
        return 0.5 * torch.mean(
            torch.log(variance) + (target - mean)**2 / variance
        )