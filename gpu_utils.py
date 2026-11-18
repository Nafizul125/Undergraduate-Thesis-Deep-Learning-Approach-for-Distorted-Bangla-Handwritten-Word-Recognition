"""
GPU Memory Management & Utilities for Pix2Pix Training
Handles CUDA detection, dynamic batch sizing, and memory monitoring
"""

import torch
import psutil
import warnings


class GPUManager:
    """Manage GPU memory and optimize batch sizes"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.device = self._detect_device()
        self.device_name = self._get_device_name()
        self.total_memory = self._get_total_memory()
        self.optimal_batch_size = None
        
        if self.verbose:
            self._print_gpu_info()
    
    def _detect_device(self):
        """Automatically detect GPU, fallback to CPU"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of VRAM
            torch.cuda.empty_cache()
            return device
        else:
            if self.verbose:
                warnings.warn("CUDA not available! Falling back to CPU. Training will be very slow.", 
                            UserWarning)
            return torch.device('cpu')
    
    def _get_device_name(self):
        """Get device name"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(0)
        return 'CPU'
    
    def _get_total_memory(self):
        """Get total memory in MB"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(0).total_memory / 1e6  # Convert to MB
        else:
            return psutil.virtual_memory().total / 1e6
    
    def _print_gpu_info(self):
        """Print GPU information"""
        print("\n" + "="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print(f"Device: {self.device_name}")
        print(f"Device Type: {self.device.type.upper()}")
        print(f"Total Memory: {self.total_memory:.0f} MB ({self.total_memory/1024:.2f} GB)")
        if self.device.type == 'cuda':
            print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print("="*70 + "\n")
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e6  # MB
            reserved = torch.cuda.memory_reserved() / 1e6    # MB
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'available_mb': (self.total_memory - reserved),
                'utilization_percent': (reserved / self.total_memory) * 100
            }
        else:
            vm = psutil.virtual_memory()
            return {
                'allocated_mb': vm.used / 1e6,
                'reserved_mb': vm.used / 1e6,
                'available_mb': vm.available / 1e6,
                'utilization_percent': vm.percent
            }
    
    def print_memory_stats(self):
        """Print memory statistics"""
        stats = self.get_memory_usage()
        print(f"\n[MEMORY] Memory Usage:")
        print(f"   Allocated: {stats['allocated_mb']:.1f} MB")
        print(f"   Reserved: {stats['reserved_mb']:.1f} MB")
        print(f"   Available: {stats['available_mb']:.1f} MB")
        print(f"   Utilization: {stats['utilization_percent']:.1f}%")
    
    def find_optimal_batch_size(self, model, input_shape, max_batch_size=64, 
                               target_memory_percent=80):
        """
        Find optimal batch size that fits in GPU memory
        
        Args:
            model: PyTorch model
            input_shape: Tuple of input dimensions (C, H, W)
            max_batch_size: Maximum batch size to test
            target_memory_percent: Target memory utilization percentage
        
        Returns:
            Optimal batch size
        """
        print("\n[SEARCH] Searching for optimal batch size...")
        
        model.to(self.device)
        model.eval()
        
        # Try different batch sizes
        tested_sizes = []
        for batch_size in [2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                stats = self.get_memory_usage()
                tested_sizes.append({
                    'batch_size': batch_size,
                    'memory_mb': stats['reserved_mb'],
                    'utilization_percent': stats['utilization_percent']
                })
                
                print(f"  [OK] Batch size {batch_size:2d}: {stats['reserved_mb']:6.0f} MB "
                      f"({stats['utilization_percent']:5.1f}%)")
                
                # Stop if we exceed target
                if stats['utilization_percent'] > target_memory_percent:
                    break
                
            except RuntimeError as e:
                print(f"  [FAIL] Batch size {batch_size:2d}: OUT OF MEMORY")
                break
        
        # Find largest batch size within target
        valid_sizes = [s for s in tested_sizes 
                      if s['utilization_percent'] <= target_memory_percent]
        
        if valid_sizes:
            optimal = valid_sizes[-1]
            self.optimal_batch_size = optimal['batch_size']
            print(f"\n[DONE] Optimal batch size: {optimal['batch_size']} "
                  f"({optimal['memory_mb']:.0f} MB, {optimal['utilization_percent']:.1f}%)")
        else:
            print(f"\n[WARN] Could not find safe batch size. Using batch size 2.")
            self.optimal_batch_size = 2
        
        torch.cuda.empty_cache()
        return self.optimal_batch_size
    
    def clear_memory(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def enable_tf32(self):
        """Enable TF32 for faster training on NVIDIA GPUs"""
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.verbose:
                print("[OK] TF32 enabled for faster computation")
    
    def disable_tf32(self):
        """Disable TF32 for exact computations"""
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


class ScalarLogger:
    """Simple scalar logging for tracking metrics"""
    
    def __init__(self):
        self.logs = {}
    
    def log(self, name, value, step):
        """Log a scalar value"""
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append({'step': step, 'value': value})
    
    def get_average(self, name, last_n=None):
        """Get average of last n values"""
        if name not in self.logs or len(self.logs[name]) == 0:
            return 0.0
        
        values = self.logs[name]
        if last_n:
            values = values[-last_n:]
        
        return sum(v['value'] for v in values) / len(values)
    
    def clear(self):
        """Clear all logs"""
        self.logs = {}


def setup_training_environment(config, verbose=True):
    """
    Setup complete training environment:
    - GPU detection and optimization
    - Memory optimization
    - Batch size optimization
    """
    gpu_manager = GPUManager(verbose=verbose)
    
    # Enable optimizations
    gpu_manager.enable_tf32()
    
    return gpu_manager


