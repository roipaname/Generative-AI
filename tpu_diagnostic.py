#!/usr/bin/env python3
"""
TPU Diagnostics Script - Check TPU configuration and connectivity
"""
import os
import sys

print("=== TPU Configuration Diagnostics ===\n")

# 1. Environment Variables
print("1. Environment Variables:")
tpu_vars = ['PJRT_DEVICE', 'TPU_NAME', 'XLA_USE_SPMD', 'COLAB_TPU_ADDR', 'TPU_WORKER_IP', 'TPU_WORKER_HOSTPORT']
for var in tpu_vars:
    value = os.environ.get(var, "NOT SET")
    print(f"   {var}: {value}")

print("\n2. System Information:")
print(f"   Python version: {sys.version}")
print(f"   Platform: {sys.platform}")

# 3. Check if running on Colab
print("\n3. Runtime Environment:")
try:
    import google.colab
    print("   Running on: Google Colab")
    print("   Colab TPU setup required")
except ImportError:
    print("   Running on: Local/Cloud VM")

# 4. TPU Detection
print("\n4. TPU Detection:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    import torch_xla
    print(f"   torch_xla version: {torch_xla.__version__}")
    
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    
    # Check for deprecated vs new API
    try:
        world_size = xr.world_size()
        print(f"   TPU cores (new API): {world_size}")
    except:
        try:
            world_size = xm.xrt_world_size()
            print(f"   TPU cores (deprecated API): {world_size}")
        except Exception as e:
            print(f"   Error getting world size: {e}")
            world_size = 0
    
    # Try to get TPU device
    try:
        device = xm.xla_device()
        print(f"   XLA device: {device}")
    except Exception as e:
        print(f"   Error getting XLA device: {e}")
    
    # Check for TPU availability
    try:
        tpu_available = xm.get_xla_supported_devices("TPU")
        print(f"   Available TPU devices: {len(tpu_available)}")
        for i, tpu in enumerate(tpu_available):
            print(f"     TPU {i}: {tpu}")
    except Exception as e:
        print(f"   Error checking TPU devices: {e}")
        
except ImportError as e:
    print(f"   torch_xla not installed: {e}")

# 5. Colab-specific TPU setup
print("\n5. Colab TPU Setup (if applicable):")
try:
    import google.colab
    print("   Colab detected - checking TPU configuration...")
    
    # Check if TPU address is set
    tpu_address = os.environ.get('COLAB_TPU_ADDR')
    if tpu_address:
        print(f"   TPU Address: {tpu_address}")
        
        # Try to connect to TPU
        try:
            import requests
            response = requests.get(f'http://{tpu_address}:8470/requestversion/tpu_driver0.1-dev20191206')
            print(f"   TPU connection test: SUCCESS (status: {response.status_code})")
        except Exception as e:
            print(f"   TPU connection test: FAILED ({e})")
    else:
        print("   No TPU address found - TPU might not be enabled")
        print("   To enable: Runtime > Change runtime type > Hardware accelerator > TPU")
        
except ImportError:
    print("   Not running on Colab")

# 6. Recommendations
print("\n6. Recommendations:")
if 'world_size' in locals() and world_size == 1:
    print("   ⚠️  Only 1 TPU core detected instead of 8")
    print("   Possible fixes:")
    print("   - If on Colab: Restart runtime and ensure TPU is selected")
    print("   - If on GCP VM: Check TPU configuration and network setup")
    print("   - Verify environment variables are set correctly")
    print("   - Try the TPU connection setup script below")
elif 'world_size' in locals() and world_size == 8:
    print("   ✅ All 8 TPU cores detected - configuration looks good!")
else:
    print("   ❌ TPU not properly detected - check installation and setup")

print("\n=== End Diagnostics ===")