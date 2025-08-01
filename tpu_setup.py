#!/usr/bin/env python3
"""
TPU Connection Setup Script
This script properly configures TPU v2-8 for multi-core access
"""
import os
import sys
import time

def setup_colab_tpu():
    """Setup TPU for Google Colab environment"""
    print("Setting up TPU for Google Colab...")
    
    try:
        import google.colab
        print("✓ Colab environment detected")
        
        # Install required packages
        import subprocess
        print("Installing cloud-tpu-client...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'cloud-tpu-client==0.10'], 
                      capture_output=True, check=True)
        
        # Get TPU address
        tpu_address = os.environ.get('COLAB_TPU_ADDR')
        if not tpu_address:
            print("❌ TPU address not found!")
            print("Please ensure TPU is enabled: Runtime > Change runtime type > Hardware accelerator > TPU")
            return False
            
        print(f"✓ TPU address found: {tpu_address}")
        
        # Configure environment
        os.environ['TPU_NAME'] = f'grpc://{tpu_address}'
        os.environ['XLA_USE_SPMD'] = '1'
        os.environ['PJRT_DEVICE'] = 'TPU'
        
        # Import and configure XLA
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Force TPU initialization
        device = xm.xla_device()
        print(f"✓ XLA device initialized: {device}")
        
        # Check world size
        world_size = xm.xrt_world_size()
        print(f"✓ TPU cores detected: {world_size}")
        
        if world_size == 8:
            print("🎉 All 8 TPU cores successfully configured!")
            return True
        else:
            print(f"⚠️ Expected 8 cores, got {world_size}")
            return False
            
    except ImportError:
        print("❌ Not running on Colab")
        return False
    except Exception as e:
        print(f"❌ Error setting up Colab TPU: {e}")
        return False

def setup_gcp_vm_tpu():
    """Setup TPU for Google Cloud VM environment"""
    print("Setting up TPU for GCP VM...")
    
    try:
        # Check if TPU environment variables are set
        tpu_name = os.environ.get('TPU_NAME')
        if not tpu_name:
            print("❌ TPU_NAME environment variable not set")
            print("Set it with: export TPU_NAME=your-tpu-name")
            return False
            
        print(f"✓ TPU_NAME: {tpu_name}")
        
        # Set required environment variables
        os.environ['XLA_USE_SPMD'] = '1'
        os.environ['PJRT_DEVICE'] = 'TPU'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Additional optimization settings
        os.environ['XLA_PERSISTENT_CACHE_DEVICE'] = '1'
        os.environ['XLA_CACHE_SIZE'] = '128MB'
        
        # Import and test TPU
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Initialize TPU
        device = xm.xla_device()
        print(f"✓ XLA device: {device}")
        
        # Check cores
        world_size = xm.xrt_world_size()
        print(f"✓ TPU cores: {world_size}")
        
        if world_size == 8:
            print("🎉 All 8 TPU cores configured!")
            return True
        else:
            print(f"⚠️ Expected 8 cores, got {world_size}")
            print("This might indicate TPU configuration issues")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up GCP VM TPU: {e}")
        return False

def test_tpu_multicore():
    """Test multi-core TPU functionality"""
    print("\nTesting multi-core TPU functionality...")
    
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        
        def test_fn():
            device = xm.xla_device()
            rank = xm.get_ordinal()
            world_size = xm.xrt_world_size()
            
            print(f"Process {rank}/{world_size} on device {device}")
            
            # Simple tensor operation
            x = torch.randn(2, 2).to(device)
            y = torch.matmul(x, x)
            
            # Synchronize
            xm.mark_step()
            return f"Rank {rank}: Success"
        
        print("Spawning test processes on all TPU cores...")
        results = xmp.spawn(test_fn, nprocs=8, start_method='spawn')
        
        print("✓ Multi-core test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-core test failed: {e}")
        return False

def main():
    print("=== TPU Multi-Core Setup ===\n")
    
    # Detect environment
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    success = False
    
    if is_colab:
        success = setup_colab_tpu()
    else:
        success = setup_gcp_vm_tpu()
    
    if success:
        print("\n✅ TPU setup successful!")
        
        # Test multi-core functionality
        if input("\nTest multi-core functionality? (y/n): ").lower() == 'y':
            test_tpu_multicore()
    else:
        print("\n❌ TPU setup failed!")
        print("\nTroubleshooting steps:")
        print("1. Ensure TPU is properly allocated and running")
        print("2. Check network connectivity to TPU")
        print("3. Verify environment variables")
        print("4. Restart runtime/kernel and try again")

if __name__ == "__main__":
    main()