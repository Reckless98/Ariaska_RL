"""
Performance optimization guidelines and utilities for ARIASKA_RL.
"""
import time
import psutil
import torch
import logging
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    throughput: float = 0.0  # operations per second


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking system resources
    and training performance.
    """
    
    def __init__(self, log_interval: int = 60):
        """
        Initialize performance monitor.
        
        Args:
            log_interval: Interval in seconds for logging metrics
        """
        self.log_interval = log_interval
        self.start_time = time.time()
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries to prevent memory bloat
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.log_interval)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        process = psutil.Process()
        
        # CPU and memory metrics
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # GPU metrics (if available)
        gpu_usage = 0.0
        gpu_memory = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                # GPU utilization requires nvidia-ml-py
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = gpu_util.gpu
                except ImportError:
                    pass  # pynvml not available
            except Exception as e:
                logger.debug(f"Error getting GPU metrics: {e}")
        
        return PerformanceMetrics(
            execution_time=time.time() - self.start_time,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        # Calculate averages
        avg_memory = np.mean([m.memory_usage_mb for m in self.metrics_history])
        avg_cpu = np.mean([m.cpu_usage_percent for m in self.metrics_history])
        avg_gpu = np.mean([m.gpu_usage_percent for m in self.metrics_history])
        max_memory = np.max([m.memory_usage_mb for m in self.metrics_history])
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'average_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'average_cpu_percent': avg_cpu,
            'average_gpu_percent': avg_gpu,
            'sample_count': len(self.metrics_history)
        }


@contextmanager
def performance_timer(operation_name: str, log_result: bool = True):
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        log_result: Whether to log the result
    
    Example:
        with performance_timer("model_forward_pass"):
            output = model(input_data)
    """
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        if log_result:
            logger.info(f"{operation_name}: {duration:.3f}s, memory Δ: {memory_delta:.1f}MB")


class TensorOptimizer:
    """Utilities for optimizing tensor operations"""
    
    @staticmethod
    def optimize_tensor_memory():
        """Optimize tensor memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    @staticmethod
    def enable_mixed_precision() -> Optional[torch.cuda.amp.GradScaler]:
        """Enable mixed precision training if available"""
        if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'GradScaler'):
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
            return scaler
        return None
    
    @staticmethod
    def optimize_dataloader_settings(num_workers: Optional[int] = None) -> Dict[str, Any]:
        """Get optimized DataLoader settings"""
        if num_workers is None:
            # Use number of CPU cores, but cap at 8 to avoid diminishing returns
            num_workers = min(psutil.cpu_count(), 8)
        
        settings = {
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': num_workers > 0,
        }
        
        logger.info(f"Optimized DataLoader settings: {settings}")
        return settings


class MemoryProfiler:
    """Memory profiling utilities"""
    
    @staticmethod
    def profile_memory_usage(func: Callable) -> Callable:
        """Decorator to profile memory usage of a function"""
        def wrapper(*args, **kwargs):
            import tracemalloc
            
            tracemalloc.start()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                current, peak = tracemalloc.get_traced_memory()
                end_memory = psutil.Process().memory_info().rss
                tracemalloc.stop()
                
                logger.info(f"Memory profile for {func.__name__}:")
                logger.info(f"  RSS: {(end_memory - start_memory) / 1024 / 1024:.1f}MB")
                logger.info(f"  Traced current: {current / 1024 / 1024:.1f}MB")
                logger.info(f"  Traced peak: {peak / 1024 / 1024:.1f}MB")
        
        return wrapper
    
    @staticmethod
    def get_memory_summary() -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        summary = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
        }
        
        if torch.cuda.is_available():
            summary.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            })
        
        return summary


class BatchSizeOptimizer:
    """Utilities for finding optimal batch sizes"""
    
    @staticmethod
    def find_optimal_batch_size(
        model: torch.nn.Module,
        input_shape: tuple,
        device: torch.device,
        max_batch_size: int = 1024,
        start_batch_size: int = 1
    ) -> int:
        """
        Find optimal batch size through binary search.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (without batch dimension)
            device: Device to run on
            max_batch_size: Maximum batch size to try
            start_batch_size: Starting batch size
            
        Returns:
            Optimal batch size
        """
        model.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        def test_batch_size(batch_size: int) -> bool:
            """Test if batch size works without OOM"""
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(batch_size, *input_shape, device=device)
                    _ = model(dummy_input)
                    del dummy_input
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    return False
                raise e
        
        # Binary search for optimal batch size
        low, high = start_batch_size, max_batch_size
        optimal_batch_size = start_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            if test_batch_size(mid):
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size


class TrainingSpeedOptimizer:
    """Utilities for optimizing training speed"""
    
    @staticmethod
    def compile_model(model: torch.nn.Module) -> torch.nn.Module:
        """Compile model for faster execution (PyTorch 2.0+)"""
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model)
                logger.info("Model compiled for faster execution")
                return compiled_model
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        return model
    
    @staticmethod
    def benchmark_forward_pass(
        model: torch.nn.Module,
        input_shape: tuple,
        batch_size: int = 32,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model forward pass performance.
        
        Returns:
            Dictionary with timing statistics
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        results = {
            'total_time': total_time,
            'avg_time_per_iteration': avg_time,
            'iterations_per_second': throughput,
            'avg_time_ms': avg_time * 1000,
        }
        
        logger.info(f"Forward pass benchmark: {avg_time*1000:.2f}ms/iter, {throughput:.1f} iter/s")
        return results


# Performance optimization recommendations
PERFORMANCE_RECOMMENDATIONS = {
    'memory': [
        "Use gradient checkpointing for large models",
        "Enable mixed precision training with autocast",
        "Use efficient data loading with multiple workers",
        "Clear GPU cache periodically with torch.cuda.empty_cache()",
        "Use in-place operations where possible",
        "Avoid transferring tensors between CPU and GPU frequently",
    ],
    'compute': [
        "Compile models with torch.compile() (PyTorch 2.0+)",
        "Use vectorized operations instead of loops",
        "Batch operations to maximize GPU utilization",
        "Use appropriate tensor data types (float16 vs float32)",
        "Enable cuDNN benchmark mode for consistent input sizes",
        "Use fused optimizers like FusedAdam",
    ],
    'io': [
        "Use DataLoader with num_workers > 0",
        "Enable pin_memory for GPU training",
        "Use memory-mapped files for large datasets",
        "Implement data caching for repeated access",
        "Use efficient file formats (HDF5, Parquet)",
        "Prefetch data to overlap compute and IO",
    ],
    'algorithm': [
        "Use gradient accumulation for large effective batch sizes",
        "Implement early stopping to avoid overtraining",
        "Use learning rate scheduling",
        "Implement proper weight initialization",
        "Use regularization techniques to improve convergence",
        "Consider model architecture optimizations",
    ]
}


def print_performance_recommendations():
    """Print performance optimization recommendations"""
    print("🚀 ARIASKA_RL Performance Optimization Recommendations")
    print("=" * 60)
    
    for category, recommendations in PERFORMANCE_RECOMMENDATIONS.items():
        print(f"\n📊 {category.upper()}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n💡 For more detailed optimization guides, see the documentation.")


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor(log_interval=5)
    monitor.start_monitoring()
    
    try:
        # Simulate some work
        time.sleep(10)
        
        # Print recommendations
        print_performance_recommendations()
        
        # Print monitoring summary
        summary = monitor.get_summary()
        print(f"\nMonitoring Summary: {summary}")
        
    finally:
        monitor.stop_monitoring()