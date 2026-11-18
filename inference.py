import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
import argparse
import os
from tqdm import tqdm
import warnings

from models import Pix2PixModel
from gpu_utils import GPUManager
import torchvision.transforms as transforms


class Pix2PixInference:
    """Enhanced inference with GPU optimization and batch processing"""
    
    def __init__(self, config_path, checkpoint_path, device=None, auto_detect=True):
        """
        Initialize inference
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            auto_detect: Auto-detect best device
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if auto_detect and device is None:
            self.gpu_manager = GPUManager(verbose=True)
            self.device = self.gpu_manager.device
        else:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)
            self.gpu_manager = GPUManager(verbose=False)
        
        # Load model
        print(f"\n📦 Loading model from {checkpoint_path}...")
        self.model = Pix2PixModel(self.config).to(self.device)
        self.model.generator.eval()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        print(f"✓ Model loaded successfully")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def _denormalize(self, img_tensor):
        """Convert from [-1, 1] to [0, 1]"""
        img_tensor = img_tensor * 0.5 + 0.5
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return img_tensor
    
    def process_image(self, input_image_path, return_arrays=False):
        """
        Process single image
        
        Args:
            input_image_path: Path to input image
            return_arrays: If True, return numpy arrays instead of PIL images
        
        Returns:
            Tuple of (input_image, output_image)
        """
        # Load image
        img = Image.open(input_image_path).convert('RGB')
        img_array = np.array(img)
        
        # Extract input (A) - left 256 pixels
        A_array = img_array[:, :256, :] if img_array.shape[1] > 256 else img_array
        A_pil = Image.fromarray(A_array).resize((256, 256), Image.BICUBIC)
        
        # Convert to tensor and add batch dimension
        A_tensor = self.transform(A_pil).unsqueeze(0).to(self.device)
        
        # Generate output
        with torch.no_grad():
            B_tensor = self.model.generator(A_tensor)
        
        # Convert output back
        B_tensor = B_tensor.squeeze(0).cpu()
        B_tensor = self._denormalize(B_tensor)
        B_pil = transforms.ToPILImage()(B_tensor)
        
        if return_arrays:
            return np.array(A_pil), np.array(B_pil)
        return A_pil, B_pil
    
    def process_batch(self, input_paths, batch_size=8):
        """
        Process multiple images efficiently
        
        Args:
            input_paths: List of input image paths
            batch_size: Batch size for processing
        
        Yields:
            Tuple of (input_path, input_image, output_image)
        """
        print(f"\n🔄 Processing {len(input_paths)} images with batch size {batch_size}...")
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(input_paths), batch_size)):
                batch_end = min(batch_start + batch_size, len(input_paths))
                batch_paths = input_paths[batch_start:batch_end]
                
                # Load batch
                batch_A = []
                batch_pils = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        img_array = np.array(img)
                        A_array = img_array[:, :256, :] if img_array.shape[1] > 256 else img_array
                        A_pil = Image.fromarray(A_array).resize((256, 256), Image.BICUBIC)
                        batch_pils.append(A_pil)
                        
                        A_tensor = self.transform(A_pil)
                        batch_A.append(A_tensor)
                    except Exception as e:
                        print(f"  ⚠️  Error loading {path}: {e}")
                        batch_A.append(None)
                
                # Skip if all failed
                if all(x is None for x in batch_A):
                    continue
                
                # Stack valid images
                valid_indices = [i for i, x in enumerate(batch_A) if x is not None]
                if valid_indices:
                    valid_tensors = [batch_A[i] for i in valid_indices]
                    batch_A_tensor = torch.stack(valid_tensors).to(self.device)
                    
                    # Generate batch
                    batch_B_tensor = self.model.generator(batch_A_tensor)
                    
                    # Convert outputs
                    for i, idx in enumerate(valid_indices):
                        B_tensor = batch_B_tensor[i].cpu()
                        B_tensor = self._denormalize(B_tensor)
                        B_pil = transforms.ToPILImage()(B_tensor)
                        
                        yield batch_paths[idx], batch_pils[idx], B_pil
    
    def process_directory(self, input_dir, output_dir, create_comparison=True):
        """
        Process all images in directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            create_comparison: If True, create side-by-side comparisons
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if create_comparison:
            comparison_dir = os.path.join(output_dir, 'comparisons')
            Path(comparison_dir).mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        print(f"\n📁 Found {len(image_files)} images in {input_dir}")
        
        # Process images
        for input_path, A_pil, B_pil in self.process_batch(image_files):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)
            
            # Save output
            B_pil.save(output_path)
            
            # Save comparison if requested
            if create_comparison:
                comp_path = os.path.join(comparison_dir, f'cmp_{filename}')
                self._save_comparison(A_pil, B_pil, comp_path)
        
        print(f"✓ Processing complete! Results saved to {output_dir}")
    
    def _save_comparison(self, img_A, img_B, output_path):
        """Save side-by-side comparison image"""
        # Ensure same size
        size = (256, 256)
        img_A = img_A.resize(size, Image.BICUBIC) if img_A.size != size else img_A
        img_B = img_B.resize(size, Image.BICUBIC) if img_B.size != size else img_B
        
        # Create comparison
        comp = Image.new('RGB', (512, 256))
        comp.paste(img_A, (0, 0))
        comp.paste(img_B, (256, 0))
        comp.save(output_path)
    
    def process_and_visualize(self, input_image_path, output_path=None, show_comparison=True):
        """
        Process image and optionally save visualization
        
        Args:
            input_image_path: Path to input image
            output_path: Output path for visualization
            show_comparison: If True, create side-by-side comparison
        """
        print(f"\n🎨 Processing {os.path.basename(input_image_path)}...")
        A, B = self.process_image(input_image_path)
        
        if show_comparison:
            self._save_comparison(A, B, output_path or input_image_path.replace('.png', '_restored.png'))
            print(f"✓ Saved to {output_path or input_image_path.replace('.png', '_restored.png')}")
        else:
            B.save(output_path or input_image_path.replace('.png', '_output.png'))
            print(f"✓ Saved to {output_path or input_image_path.replace('.png', '_output.png')}")
        
        return A, B
    
    def benchmark(self, num_runs=10):
        """Benchmark inference speed"""
        print(f"\n⏱️  Benchmarking inference speed ({num_runs} runs)...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generator(dummy_input)
        
        # Benchmark
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                
                _ = self.model.generator(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])  # Skip first run
        fps = 1 / avg_time
        
        print(f"✓ Average time: {avg_time*1000:.2f} ms")
        print(f"✓ FPS: {fps:.2f}")
        print(f"✓ Device: {self.device}")


def main():
    parser = argparse.ArgumentParser(description='Pix2Pix Inference - Bangla Handwriting Restoration')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint (e.g., checkpoints/checkpoint_epoch_010.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory or file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda, cpu, or auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing multiple images')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Do not create side-by-side comparisons')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark instead of processing')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = Pix2PixInference(
        args.config, 
        args.checkpoint, 
        device=args.device,
        auto_detect=True
    )
    
    if args.benchmark:
        # Run benchmark
        inference.benchmark(num_runs=10)
    elif os.path.isdir(args.input):
        # Process directory
        output_dir = args.output or os.path.join(args.input, 'results')
        inference.process_directory(
            args.input, 
            output_dir,
            create_comparison=not args.no_comparison
        )
    else:
        # Process single image
        output_file = args.output or args.input.replace('.png', '_restored.png')
        inference.process_and_visualize(
            args.input, 
            output_file,
            show_comparison=not args.no_comparison
        )


if __name__ == '__main__':
    main()

