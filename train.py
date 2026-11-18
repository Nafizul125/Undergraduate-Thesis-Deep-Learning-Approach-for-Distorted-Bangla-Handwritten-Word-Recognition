import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
from pathlib import Path
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import gc

from dataset import get_dataloaders
from models import Pix2PixModel
from gpu_utils import GPUManager, setup_training_environment, ScalarLogger


class PerceptualLoss(nn.Module):
    """VGG19 Perceptual Loss for preserving handwriting style"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Load pre-trained VGG19
        vgg19 = models.vgg19(pretrained=True)
        # Extract features up to layer 36 (before classifier)
        self.features = nn.Sequential(*list(vgg19.features.children())[:36]).to(device)
        # Freeze VGG19 weights
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()
        
        # Normalization values for ImageNet (create on device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        # Normalize for VGG19
        x_norm = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y_norm = (y + 1) / 2
        x_norm = (x_norm - self.mean) / self.std
        y_norm = (y_norm - self.mean) / self.std
        
        # Extract features
        x_feat = self.features(x_norm)
        y_feat = self.features(y_norm)
        
        # MSE loss on features
        loss = nn.functional.mse_loss(x_feat, y_feat)
        return loss


class Pix2PixTrainer:
    def __init__(self, config_path, resume_from=None, auto_batch_size=True):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup GPU
        self.gpu_manager = setup_training_environment(self.config)
        self.device = self.gpu_manager.device
        
        # Create directories
        Path(self.config['logging']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['logging']['tensorboard_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['logging']['sample_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = Pix2PixModel(self.config).to(self.device)
        
        # Count parameters
        gen_params = sum(p.numel() for p in self.model.generator.parameters())
        dis_params = sum(p.numel() for p in self.model.discriminator.parameters())
        print(f"\n[MODEL PARAMETERS]")
        print(f"   Generator: {gen_params/1e6:.2f}M")
        print(f"   Discriminator: {dis_params/1e6:.2f}M")
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss(self.device).to(self.device)
        
        # Optimizers
        lr_gen = self.config['training']['learning_rate_gen']
        lr_dis = self.config['training']['learning_rate_dis']
        beta1 = self.config['training']['beta1']
        
        self.optimizer_gen = Adam(
            self.model.generator.parameters(),
            lr=lr_gen,
            betas=(beta1, 0.999)
        )
        
        self.optimizer_dis = Adam(
            self.model.discriminator.parameters(),
            lr=lr_dis,
            betas=(beta1, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_gen = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_gen,
            lr_lambda=lambda epoch: 1.0 if epoch < 100 else 0.5
        )
        
        self.scheduler_dis = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_dis,
            lr_lambda=lambda epoch: 1.0 if epoch < 100 else 0.5
        )
        
        # Mixed precision training
        self.use_amp = self.config['training'].get('use_mixed_precision', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("[OK] Mixed precision training enabled")
        
        # TensorBoard
        self.writer = SummaryWriter(self.config['logging']['tensorboard_dir'])
        
        # Logger for metrics
        self.metric_logger = ScalarLogger()
        
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Batch size optimization
        if auto_batch_size and self.device.type == 'cuda':
            print("\n[INFO] Optimizing batch size for your GPU...")
            try:
                optimal_bs = self.gpu_manager.find_optimal_batch_size(
                    self.model,
                    input_shape=(3, 256, 256),
                    max_batch_size=self.config['dataset']['batch_size'],
                    target_memory_percent=80
                )
                self.config['dataset']['batch_size'] = optimal_bs
                print(f"[OK] Batch size updated: {optimal_bs}")
            except Exception as e:
                print(f"[WARN] Could not optimize batch size: {str(e)}")
                print(f"   Using configured batch size: {self.config['dataset']['batch_size']}")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model and optimizer states"""
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[OK] Resumed from epoch {self.start_epoch}")
        print(f"[OK] Best val loss so far: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch):
        """Save model and optimizer states"""
        checkpoint_path = os.path.join(
            self.config['logging']['checkpoint_dir'],
            f"checkpoint_epoch_{epoch:03d}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
            'optimizer_dis_state_dict': self.optimizer_dis.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)

        print(f"[OK] Checkpoint saved: {checkpoint_path}")

    def save_best_model(self, epoch, val_loss):
        """Save model only when validation loss improves."""
        best_path = os.path.join(
            self.config['logging']['checkpoint_dir'], 'best_model.pth'
        )
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
            'optimizer_dis_state_dict': self.optimizer_dis.state_dict(),
            'best_val_loss': val_loss,
        }, best_path)
        print(f"   [BEST] New best model saved! Val L1: {val_loss:.4f}  →  {best_path}")
    
    def generate_samples(self, batch, epoch):
        """Generate sample outputs and save comparison images"""
        self.model.generator.eval()
        
        sample_dir = self.config['logging']['sample_dir']
        epoch_dir = os.path.join(sample_dir, f'epoch_{epoch:03d}')
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            A = batch['A'][:4].to(self.device)  # Use first 4 samples
            B_real = batch['B'][:4].to(self.device)
            B_fake = self.model.generator(A)
        
        # Inverse normalization transform
        def denormalize(img):
            """Convert from [-1, 1] to [0, 1]"""
            img = img * 0.5 + 0.5
            img = torch.clamp(img, 0, 1)
            return img
        
        # Save sample comparison images
        for i in range(A.shape[0]):
            fig = Image.new('RGB', (768, 256))
            
            # Input (damaged)
            input_img = denormalize(A[i]).cpu().permute(1, 2, 0).numpy()
            input_img = (input_img * 255).astype(np.uint8)
            fig.paste(Image.fromarray(input_img), (0, 0))
            
            # Generated (reconstructed)
            gen_img = denormalize(B_fake[i]).cpu().permute(1, 2, 0).numpy()
            gen_img = (gen_img * 255).astype(np.uint8)
            fig.paste(Image.fromarray(gen_img), (256, 0))
            
            # Ground truth (clean)
            real_img = denormalize(B_real[i]).cpu().permute(1, 2, 0).numpy()
            real_img = (real_img * 255).astype(np.uint8)
            fig.paste(Image.fromarray(real_img), (512, 0))
            
            # Save with labels
            fig.save(os.path.join(epoch_dir, f'sample_{i:02d}.png'))
        
        self.model.generator.train()
        print(f"   [OK] Saved {A.shape[0]} sample comparisons to {epoch_dir}")
        print(f"     Layout: [Input | Generated | Ground Truth]")
    
    def train_step(self, batch, global_step):
        """Single training step with mixed precision"""
        A = batch['A'].to(self.device)
        B = batch['B'].to(self.device)
        batch_size = A.size(0)

        lambda_l1 = self.config['training']['lambda_l1']
        lambda_gan = self.config['training']['lambda_gan']
        lambda_perceptual = self.config['training'].get('lambda_perceptual', 10)
        dis_train_freq = self.config['training'].get('dis_train_freq', 1)

        try:
            # ==================
            # Train Generator
            # ==================
            self.optimizer_gen.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    # Single forward pass — reused by discriminator below
                    fake_B = self.model.generator(A)

                    fake_AB = torch.cat([A, fake_B], dim=1)
                    pred_fake = self.model.discriminator(fake_AB)

                    # GAN loss with label smoothing (0.9 for real)
                    loss_gan = self.criterion_gan(
                        pred_fake, torch.ones_like(pred_fake) * 0.9
                    )
                    loss_l1 = self.criterion_l1(fake_B, B)
                    loss_perceptual = self.criterion_perceptual(fake_B, B)
                    loss_gen = (lambda_gan * loss_gan
                                + lambda_l1 * loss_l1
                                + lambda_perceptual * loss_perceptual)

                self.scaler.scale(loss_gen).backward()
                self.scaler.step(self.optimizer_gen)
                # NOTE: scaler.update() is called once at the END of this step
            else:
                fake_B = self.model.generator(A)

                fake_AB = torch.cat([A, fake_B], dim=1)
                pred_fake = self.model.discriminator(fake_AB)

                loss_gan = self.criterion_gan(
                    pred_fake, torch.ones_like(pred_fake) * 0.9
                )
                loss_l1 = self.criterion_l1(fake_B, B)
                loss_perceptual = self.criterion_perceptual(fake_B, B)
                loss_gen = (lambda_gan * loss_gan
                            + lambda_l1 * loss_l1
                            + lambda_perceptual * loss_perceptual)

                loss_gen.backward()
                self.optimizer_gen.step()

            # Detach fake_B ONCE — reuse for discriminator (no second forward pass)
            fake_B_detached = fake_B.detach()

            # ==================
            # Train Discriminator (optionally less frequently to stabilize)
            # ==================
            if global_step % dis_train_freq == 0:
                self.optimizer_dis.zero_grad()

                if self.use_amp:
                    with autocast('cuda'):
                        # Real pair (label smoothing: 0.9)
                        real_AB = torch.cat([A, B], dim=1)
                        pred_real = self.model.discriminator(real_AB)
                        loss_dis_real = self.criterion_gan(
                            pred_real, torch.ones_like(pred_real) * 0.9
                        )

                        # Fake pair — use stored fake_B, label smoothing 0.1
                        # FIX: was "zeros * 0.1" which always equals 0
                        fake_AB_dis = torch.cat([A, fake_B_detached], dim=1)
                        pred_fake_dis = self.model.discriminator(fake_AB_dis)
                        loss_dis_fake = self.criterion_gan(
                            pred_fake_dis, torch.full_like(pred_fake_dis, 0.1)
                        )

                        loss_dis = (loss_dis_real + loss_dis_fake) * 0.5

                    self.scaler.scale(loss_dis).backward()
                    self.scaler.step(self.optimizer_dis)
                else:
                    real_AB = torch.cat([A, B], dim=1)
                    pred_real = self.model.discriminator(real_AB)
                    loss_dis_real = self.criterion_gan(
                        pred_real, torch.ones_like(pred_real) * 0.9
                    )

                    fake_AB_dis = torch.cat([A, fake_B_detached], dim=1)
                    pred_fake_dis = self.model.discriminator(fake_AB_dis)
                    loss_dis_fake = self.criterion_gan(
                        pred_fake_dis, torch.full_like(pred_fake_dis, 0.1)
                    )

                    loss_dis = (loss_dis_real + loss_dis_fake) * 0.5
                    loss_dis.backward()
                    self.optimizer_dis.step()
            else:
                loss_dis = torch.tensor(0.0)

            # Call scaler.update() ONCE per training step (covers both gen and dis)
            if self.use_amp:
                self.scaler.update()
            
            # Logging
            if global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Loss/Generator', loss_gen.item(), global_step)
                self.writer.add_scalar('Loss/Discriminator', loss_dis.item(), global_step)
                self.writer.add_scalar('Loss/L1', loss_l1.item(), global_step)
                self.writer.add_scalar('Loss/GAN', loss_gan.item(), global_step)
                self.writer.add_scalar('Loss/Perceptual', loss_perceptual.item(), global_step)
                
                # GPU memory logging
                if self.device.type == 'cuda':
                    gpu_stats = self.gpu_manager.get_memory_usage()
                    self.writer.add_scalar('GPU/Memory_MB', gpu_stats['reserved_mb'], global_step)
                    self.writer.add_scalar('GPU/Utilization_%', gpu_stats['utilization_percent'], global_step)
            
            # Log metrics
            self.metric_logger.log('loss_gen', loss_gen.item(), global_step)
            self.metric_logger.log('loss_dis', loss_dis.item(), global_step)
            
            return {
                'loss_gen': loss_gen.item(),
                'loss_dis': loss_dis.item(),
                'loss_l1': loss_l1.item(),
                'loss_gan': loss_gan.item(),
                'loss_perceptual': loss_perceptual.item(),
            }
        
        except RuntimeError as e:
            print(f"\n[ERROR] CUDA Out of Memory Error!")
            print(f"   Error: {str(e)}")
            print(f"   Try reducing batch size in config.yaml")
            self.gpu_manager.clear_memory()
            gc.collect()
            raise
    
    def validate(self, val_loader, epoch):
        """Validation step"""
        self.model.generator.eval()
        
        total_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                A = batch['A'].to(self.device)
                B = batch['B'].to(self.device)
                
                if self.use_amp:
                    with autocast('cuda'):
                        fake_B = self.model.generator(A)
                        loss_l1 = self.criterion_l1(fake_B, B)
                else:
                    fake_B = self.model.generator(A)
                    loss_l1 = self.criterion_l1(fake_B, B)
                
                total_l1 += loss_l1.item() * A.size(0)
        
        avg_l1 = total_l1 / len(val_loader.dataset)
        self.writer.add_scalar('Validation/L1_Loss', avg_l1, epoch)
        
        self.model.generator.train()
        
        print(f"   [OK] Validation L1 Loss: {avg_l1:.4f}")
        return avg_l1
    
    def train(self, num_epochs=None, resume_from=None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        # Create dataloaders
        print("\n[LOADING] Loading datasets...")
        train_loader, val_loader = get_dataloaders(self.config)
        
        print(f"\n{'='*70}")
        print("[TRAINING] CONFIGURATION")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device} ({self.gpu_manager.device_name})")
        print(f"Batch Size: {self.config['dataset']['batch_size']}")
        print(f"Training Samples: {len(train_loader.dataset)}")
        print(f"Validation Samples: {len(val_loader.dataset)}")
        print(f"Mixed Precision: {'Yes' if self.use_amp else 'No'}")
        print(f"{'='*70}\n")
        
        self.model.generator.train()
        self.model.discriminator.train()
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n[EPOCH] Epoch [{epoch + 1}/{num_epochs}]")
            
            pbar = tqdm(train_loader, desc="Training", leave=True)
            epoch_losses = {'gen': 0, 'dis': 0, 'l1': 0, 'gan': 0, 'perceptual': 0}
            
            for batch in pbar:
                losses = self.train_step(batch, self.global_step)
                
                epoch_losses['gen'] += losses['loss_gen']
                epoch_losses['dis'] += losses['loss_dis']
                epoch_losses['l1'] += losses['loss_l1']
                epoch_losses['gan'] += losses['loss_gan']
                epoch_losses['perceptual'] += losses['loss_perceptual']
                
                pbar.set_postfix({
                    'Gen': f"{losses['loss_gen']:.3f}",
                    'Dis': f"{losses['loss_dis']:.3f}",
                    'L1': f"{losses['loss_l1']:.3f}",
                })
                
                self.global_step += 1
            
            # Average losses
            num_batches = len(train_loader)
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            
            print(f"   [LOSS] Gen: {avg_losses['gen']:.4f} | Dis: {avg_losses['dis']:.4f}")
            print(f"   [LOSS] L1: {avg_losses['l1']:.4f} | GAN: {avg_losses['gan']:.4f}")
            print(f"   [LOSS] Perceptual: {avg_losses['perceptual']:.4f}")
            
            # GPU memory stats
            if self.device.type == 'cuda':
                gpu_stats = self.gpu_manager.get_memory_usage()
                print(f"   [GPU] Memory: {gpu_stats['reserved_mb']:.0f}/{self.gpu_manager.total_memory:.0f} MB "
                      f"({gpu_stats['utilization_percent']:.1f}%)")
            
            # Generate sample outputs
            if (epoch + 1) % self.config['logging'].get('sample_interval', 1) == 0:
                print(f"   [SAMPLES] Generating samples...")
                try:
                    sample_batch = next(iter(train_loader))
                    self.generate_samples(sample_batch, epoch + 1)
                except Exception as e:
                    print(f"   [WARN] Could not generate samples: {str(e)}")
            
            # Validation
            val_interval = self.config['training'].get('val_interval', 1)
            if val_interval > 0 and (epoch + 1) % val_interval == 0:
                print(f"   [VALIDATION] Validating...")
                val_loss = self.validate(val_loader, epoch + 1)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_best_model(epoch, val_loss)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Learning rate scheduling
            self.scheduler_gen.step()
            self.scheduler_dis.step()
        
        self.writer.close()
        print(f"\n{'='*70}")
        print("[OK] Training completed!")
        print(f"{'='*70}\n")



def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model for Bangla Handwriting Restoration')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--no-auto-batch-size', action='store_true',
                       help='Disable automatic batch size optimization')
    
    args = parser.parse_args()
    
    trainer = Pix2PixTrainer(
        args.config, 
        resume_from=args.resume,
        auto_batch_size=not args.no_auto_batch_size
    )
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
