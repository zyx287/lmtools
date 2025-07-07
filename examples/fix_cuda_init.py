'''
Utility script to help diagnose and fix CUDA initialization issues
'''
import os
import sys
import subprocess

def check_cuda_setup():
    '''Check CUDA setup and provide recommendations'''
    print("CUDA Diagnostic Tool for lmtools.seg.cellpose_segmentation")
    print("=" * 60)
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"\nEnvironment:")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check if torch is already imported
    if 'torch' in sys.modules:
        print("\n⚠️  WARNING: torch is already imported!")
        print("  This is OK for the updated cellpose_segmentation module,")
        print("  which now uses torch.cuda.set_device() instead of env vars.")
    
    # Try to import torch and check CUDA
    try:
        import torch
        print(f"\nPyTorch Info:")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print("\nGPU Details:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({mem_gb:.2f} GB)")
                
                # Check if we can switch to this device
                try:
                    torch.cuda.set_device(i)
                    print(f"         ✓ Can switch to this device")
                except Exception as e:
                    print(f"         ✗ Cannot switch to this device: {e}")
    except Exception as e:
        print(f"\n❌ Error checking PyTorch/CUDA: {e}")
    
    # Check nvidia-smi
    print("\nnvidia-smi Info:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    idx, name, total, free = parts[:4]
                    print(f"  GPU {idx}: {name} (Total: {total}, Free: {free})")
        else:
            print("  nvidia-smi not found or failed")
    except Exception as e:
        print(f"  Error running nvidia-smi: {e}")
    
    print("\n" + "=" * 60)
    print("Resolution for 'CUDA unknown error':")
    print("\n✅ The cellpose_segmentation module has been updated to fix this issue!")
    print("   It now uses torch.cuda.set_device() instead of manipulating CUDA_VISIBLE_DEVICES")
    print("\nFor single-GPU mode:")
    print("  - The module will automatically use the available GPU")
    print("  - No special environment setup needed")
    print("\nFor multi-GPU mode:")
    print("  - Each subprocess will use torch.cuda.set_device(gpu_id)")
    print("  - All GPUs remain visible, but each process uses only its assigned GPU")
    print("\nIf you still see errors:")
    print("  1. Make sure you're using the updated cellpose_segmentation.py")
    print("  2. Check that your GPUs are not already in use by other processes")
    print("  3. Try restarting your Python session")
    print("  4. Optionally set CUDA_VISIBLE_DEVICES before starting Python:")
    print("     export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0")

if __name__ == "__main__":
    check_cuda_setup()