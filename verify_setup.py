"""
🔍 Pix2Pix Setup Verification Script
Checks all components and GPU setup
"""

import sys
import importlib
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_python():
    """Check Python version"""
    print("🐍 Python Check")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ FAILED: Python 3.8+ required")
        return False
    
    print("   ✓ PASSED")
    return True


def check_packages():
    """Check required packages"""
    print("\n📦 Package Check")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
        'psutil': 'psutil',
    }
    
    all_ok = True
    for module_name, display_name in required.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✓ {display_name:20} {version}")
        except ImportError:
            print(f"   ❌ {display_name:20} NOT FOUND")
            all_ok = False
    
    if not all_ok:
        print("\n   💡 Install missing packages:")
        print("      pip install -r requirements.txt")
    
    return all_ok


def check_gpu():
    """Check GPU availability"""
    print("\n🎮 GPU Check")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✓ CUDA Available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            total_mem = props.total_memory / 1e9
            print(f"   Memory: {total_mem:.1f} GB")
            
            compute_cap = props.major, props.minor
            print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            # Estimate batch size
            estimated_batch = min(2 ** (int(total_mem / 4) + 3), 64)
            print(f"   Estimated batch size: {estimated_batch}")
            
        else:
            print("   ⚠️  CUDA Not Available (will use CPU)")
            print("   💡 Training will be very slow on CPU")
            print("   💡 Install CUDA and PyTorch with GPU support")
    
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return False
    
    return True


def check_files():
    """Check project files"""
    print("\n📁 Project Files Check")
    
    required_files = {
        'train.py': 'Training script',
        'inference.py': 'Inference script',
        'models.py': 'Model definitions',
        'dataset.py': 'Data loading',
        'gpu_utils.py': 'GPU utilities',
        'config.yaml': 'Configuration',
        'requirements.txt': 'Dependencies',
    }
    
    all_ok = True
    for filename, description in required_files.items():
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"   ✓ {filename:20} ({size:6.1f} KB) - {description}")
        else:
            print(f"   ❌ {filename:20} NOT FOUND")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check required directories"""
    print("\n📂 Directory Check")
    
    directories = {
        'data': 'Dataset root',
        'data/train': 'Training data',
        'data/val': 'Validation data',
        'checkpoints': 'Checkpoints (auto-created)',
        'runs': 'TensorBoard logs (auto-created)',
        'samples': 'Generated samples (auto-created)',
    }
    
    for dirname, description in directories.items():
        path = Path(dirname)
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob('*')))
                print(f"   ✓ {dirname:20} ({count:4} items) - {description}")
            else:
                print(f"   ❌ {dirname:20} is not a directory")
        else:
            print(f"   ⚠️  {dirname:20} missing - {description}")
    
    # Check for training data
    train_path = Path('data/train')
    val_path = Path('data/val')
    
    if train_path.exists():
        train_images = len(list(train_path.glob('*.png')))
        val_images = len(list(val_path.glob('*.png'))) if val_path.exists() else 0
        
        print(f"\n   Dataset Summary:")
        print(f"   - Training images: {train_images}")
        print(f"   - Validation images: {val_images}")
        
        if train_images < 100:
            print(f"   ⚠️  Warning: Training set < 100 images (very small)")
        
        return train_images > 0
    else:
        print(f"\n   ⚠️  No training data found!")
        print(f"   💡 Add images to: data/train/ directory")
        return False


def check_model_loading():
    """Check if models can be loaded"""
    print("\n🤖 Model Loading Check")
    
    try:
        from models import Pix2PixModel, Generator, Discriminator
        import torch
        import yaml
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("   ✓ Config loaded")
        
        # Create model
        model = Pix2PixModel(config)
        print("   ✓ Model created")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"   ✓ Model moved to {device}")
        
        # Count parameters
        gen_params = sum(p.numel() for p in model.generator.parameters())
        dis_params = sum(p.numel() for p in model.discriminator.parameters())
        print(f"   Generator: {gen_params/1e6:.2f}M parameters")
        print(f"   Discriminator: {dis_params/1e6:.2f}M parameters")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_loading():
    """Check if dataset can be loaded"""
    print("\n📚 Dataset Loading Check")
    
    try:
        from dataset import get_dataloaders
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Reduce batch size for testing
        config['dataset']['batch_size'] = 2
        
        print("   Loading data...")
        train_loader, val_loader = get_dataloaders(config)
        
        print(f"   ✓ Training loader: {len(train_loader.dataset)} images")
        print(f"   ✓ Validation loader: {len(val_loader.dataset)} images")
        
        # Try one batch
        batch = next(iter(train_loader))
        print(f"   ✓ Sample batch shapes:")
        print(f"     - Input (A): {batch['A'].shape}")
        print(f"     - Target (B): {batch['B'].shape}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu_utils():
    """Check GPU utilities"""
    print("\n⚡ GPU Utilities Check")
    
    try:
        from gpu_utils import GPUManager, setup_training_environment
        
        print("   ✓ GPU utilities imported")
        
        # Create manager (verbose=False to avoid extra output)
        manager = GPUManager(verbose=False)
        print(f"   ✓ Device: {manager.device}")
        
        # Check memory
        stats = manager.get_memory_usage()
        print(f"   ✓ Memory allocated: {stats['allocated_mb']:.1f} MB")
        print(f"   ✓ Memory available: {stats['available_mb']:.1f} MB")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error checking GPU utilities: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of all checks"""
    print_header("✅ VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks Passed: {passed}/{total}\n")
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {name}")
    
    print()
    if passed == total:
        print("  🎉 All checks passed! Ready to train!")
        print("\n  Next steps:")
        print("    1. python train.py --config config.yaml")
        print("    2. tensorboard --logdir=./runs  (in new terminal)")
        print("    3. Monitor at http://localhost:6006")
        return True
    else:
        print("  ⚠️  Some checks failed. Fix issues above before training.")
        return False


def main():
    print_header("🔍 Pix2Pix Setup Verification")
    
    results = {
        'Python 3.8+': check_python(),
        'Required Packages': check_packages(),
        'GPU/CUDA': check_gpu(),
        'Project Files': check_files(),
        'Directories': check_directories(),
        'Model Loading': check_model_loading(),
        'Dataset Loading': check_dataset_loading(),
        'GPU Utilities': check_gpu_utils(),
    }
    
    success = print_summary(results)
    
    if not success:
        print("\n  💡 For help, see:")
        print("     - TRAINING_GUIDE.md")
        print("     - ENHANCEMENTS.md")
        print("     - QUICK_START.md")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
