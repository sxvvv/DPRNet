import os
import torch
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as tfs
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataclasses import dataclass
from typing import List, Tuple

import lpips
from Depth_Anything.depth_anything.dpt import DepthAnything
from loss.CL1 import PSNRLoss
from model import ImageRestorationModel
from utils.Allweather import *
from utils.AGAN_data import *
from utils.imgqual_utils import batch_PSNR, batch_SSIM
from utils.save_image import save_colormapped_image

@dataclass
class ModelConfig:
    img_channel: int = 3
    out_channel: int = 3
    width: int = 32
    middle_blk_num: int = 1
    enc_blk_nums: List[int] = (1, 1, 1, 18)  
    dec_blk_nums: List[int] = (1, 1, 1, 1) 

@dataclass
class TrainerConfig:
    max_steps: int = 200000  
    max_epochs: int = 185
    train_type: bool = True

@dataclass
class OptimConfig:
    lr: float = 2e-4  

@dataclass
class DataConfig:
    dataset: str = "CDD-11"  
    data_dir: str = "/path/to/CDD-11_train"  
    val_data_dir: str = "/path/to/CDD-11_test"  
    image_size: int = 512  
    num_workers: int = 8

@dataclass
class TrainingConfig:
    batch_size: int = 16  

@dataclass
class SamplingConfig:
    batch_size: int = 20

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    Trainer: TrainerConfig = TrainerConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    sampling: SamplingConfig = SamplingConfig()
    image_folder: str = "/path/to/results"  # Set your results path here

class ImageRestorationTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize loss function
        self.loss_psnr = PSNRLoss()
        
        # Initialize main model
        # Updated parameters to match paper specifications
        self.model = ImageRestorationModel(
            img_channel=config.model.img_channel,
            out_channel=config.model.out_channel,
            width=config.model.width,
            middle_blk_num=config.model.middle_blk_num,
            enc_blk_nums=config.model.enc_blk_nums,
            dec_blk_nums=config.model.dec_blk_nums,
            prompt_pool=True,
            pool_size=30,  
            top_k=5,  
            diversity_weight=0.5,  
            history_size=100  
        )
        
        # Initialize DepthAnything
        self.depth = DepthAnything.from_pretrained(
            '/path/to/depth_anything_vits14',  # Set your model path here
            local_files_only=True
        ).eval()

        # Setup training parameters
        self.save_path = self.config.image_folder
        self.max_steps = self.config.Trainer.max_steps
        self.epochs = self.config.Trainer.max_epochs
        
        # Define degradation types for the CDD-11 dataset as per paper
        self.degradation_types = [
            'haze', 'rain', 'snow', 'low',  # Single degradations
            'haze_rain', 'haze_snow', 'low_haze', 'low_rain', 'low_snow', # Double degradations
            'low_haze_rain', 'low_haze_snow'  # Triple degradations
        ]
        
        # Create result directories
        self._create_result_directories()
        
        # Initialize metrics
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.automatic_optimization = True
        self.save_hyperparameters()

    def _create_result_directories(self):
        for deg_type in self.degradation_types:
            deg_path = os.path.join(self.save_path, deg_type)
            os.makedirs(deg_path, exist_ok=True)

    def closest_multiple_of_14(self, n):
        # Helper function for depth model input sizing
        return round(n / 14.0) * 14

    def lpips_score_fn(self, x, gt):
        self.lpips_fn.to(self.device)
        x = x.to(self.device)
        gt = gt.to(self.device)
        lp_score = self.lpips_fn(gt * 2 - 1, x * 2 - 1)
        return torch.mean(lp_score).item()

    def configure_optimizers(self):
        # Configure AdamW optimizer as per paper
        parameters = [{'params': self.model.parameters()}]
        optimizer = torch.optim.AdamW(parameters, lr=self.config.optim.lr)
        
        # Cosine annealing learning rate schedule as per paper
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps//len(self.train_dataloader()),
            eta_min=self.config.optim.lr * 1e-2
        )
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='psnr',
            filename='DPRNet-epoch{epoch:02d}-PSNR-{psnr:.3f}-SSIM-{ssim:.4f}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=6,
            mode="max",
            save_last=True
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [checkpoint_callback, lr_monitor]

    def training_step(self, batch, batch_idx):
        x, gt, info = batch  # Include info for degradation-specific tracking
        
        # Get depth features from Depth-Anything
        target_height = self.closest_multiple_of_14(x.shape[2])
        target_width = self.closest_multiple_of_14(x.shape[3])
        depth, features = self.depth(tfs.Resize([target_height, target_width])(x))
        
        # Forward pass with depth features as per the paper's depth-guided mechanism
        output, prompt_info = self.model(x, depth_feature=features[3][0])
        restoration_loss = self.loss_psnr(output, gt)
        
        # Calculate prompt-related loss for diversity optimization
        prompt_diversity_loss = prompt_info.get('diversity_loss', 0.0)
        
        # Calculate total loss using the composite loss function from paper
        # L_total = L_restoration + λ_p * L_prompt
        lambda_p = 0.1  # Diversity weight parameter
        total_loss = restoration_loss + lambda_p * prompt_diversity_loss
        
        # Calculate training metrics
        psnr = batch_PSNR(output.detach().float(), gt.float(), ycbcr=True)
        
        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_psnr", psnr, prog_bar=True)
        self.log("prompt_diversity", prompt_diversity_loss, prog_bar=False)
        
        # Track degradation-specific metrics if available
        if 'degradation_type' in info:
            for deg_type in info['degradation_type']:
                if deg_type in self.degradation_types:
                    self.log(f"train_psnr_{deg_type}", psnr, add_dataloader_idx=False)
        
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        input_x, target, info = batch
        
        # Get depth features
        target_height = self.closest_multiple_of_14(input_x.shape[2])
        target_width = self.closest_multiple_of_14(input_x.shape[3])
        depth, features = self.depth(tfs.Resize([target_height, target_width])(input_x))
        
        # Forward pass
        output, prompt_info = self.model(input_x, depth_feature=features[3][0])
        
        # Save images during validation
        if batch_idx == 0:
            # Process entire batch
            for idx in range(min(5, input_x.size(0))):
                # Get degradation type if available
                deg_type = "unknown"
                if 'degradation_type' in info:
                    deg_type = info['degradation_type'][idx] if idx < len(info['degradation_type']) else "unknown"
                
                # Create filenames with degradation type
                output_dir = os.path.join(self.save_path, deg_type)
                os.makedirs(output_dir, exist_ok=True)
                
                restored_filename = f"{deg_type}_restored_e{self.current_epoch}_{idx}.png"
                input_filename = f"{deg_type}_input_e{self.current_epoch}_{idx}.png"
                target_filename = f"{deg_type}_target_e{self.current_epoch}_{idx}.png"
                
                # Save images
                save_image(output[idx:idx+1], os.path.join(output_dir, restored_filename))
                save_image(input_x[idx:idx+1], os.path.join(output_dir, input_filename))
                save_image(target[idx:idx+1], os.path.join(output_dir, target_filename))
                
                # Save depth map visualization (optional)
                if depth is not None:
                    depth_filename = f"{deg_type}_depth_e{self.current_epoch}_{idx}.png"
                    save_colormapped_image(depth[idx:idx+1], os.path.join(output_dir, depth_filename))
        
        # Calculate overall metrics
        psnr = batch_PSNR(output.float(), target.float(), ycbcr=True)
        ssim = batch_SSIM(output.float(), target.float(), ycbcr=True)
        lpips_score = self.lpips_score_fn(output.float(), target.float())
        
        # Log metrics
        self.log('psnr', psnr, sync_dist=True)
        self.log('ssim', ssim, sync_dist=True)
        self.log('lpips', lpips_score)
        
        # Log degradation-specific metrics
        if 'degradation_type' in info:
            for deg_type in self.degradation_types:
                # Find samples with this degradation type
                mask = [d == deg_type for d in info['degradation_type']]
                if any(mask):
                    # Calculate metrics for this degradation type
                    idx = [i for i, m in enumerate(mask) if m]
                    deg_psnr = batch_PSNR(output[idx].float(), target[idx].float(), ycbcr=True)
                    deg_ssim = batch_SSIM(output[idx].float(), target[idx].float(), ycbcr=True)
                    
                    # Log metrics
                    self.log(f'psnr_{deg_type}', deg_psnr, sync_dist=True)
                    self.log(f'ssim_{deg_type}', deg_ssim, sync_dist=True)
        
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_score
        }

    def train_dataloader(self):
        # CDD-11 dataset with augmentations as described in the paper
        train_set = AllWeather(
            self.config.data.data_dir,
            train=True,
            size=self.config.data.image_size,
            crop=True,
            augment=True  # Enable augmentations as per paper
        )
        return DataLoader(
            train_set,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        val_set = None
        if self.config.data.dataset == 'CDD-11':
            val_set = AllWeather(self.config.data.val_data_dir, train=False, size=256, crop=False)
        elif self.config.data.dataset == 'Snow100k-S' or self.config.data.dataset == 'Snow100k-L':
            val_set = Snow100kTest(self.config.data.val_data_dir, train=False, size=256, crop=False)
        elif self.config.data.dataset == 'RESIDE-OTS':
            val_set = RESIDE_OTS(self.config.data.val_data_dir, train=False, size=256, crop=False)
        elif self.config.data.dataset == 'Rain1200':
            val_set = Rain1200(self.config.data.val_data_dir, train=False, size=256, crop=False)
            
        return DataLoader(
            val_set,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

def main():
    # Create configuration
    config = Config()
    
    # Initialize trainer
    model = ImageRestorationTrainer(config)
    
    # Setup PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_steps=config.Trainer.max_steps,
        max_epochs=config.Trainer.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir='/path/to/logs'  # Set your logs path here
    )
    
    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    main()

