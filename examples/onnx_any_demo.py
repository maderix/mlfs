#!/usr/bin/env python3
"""
ONNX Any Demo for MLFS
Shows how to mount any ONNX model using MLFS
"""

import os
import sys
import tempfile
import subprocess
import shutil
import time
from pathlib import Path

try:
    import onnx
except ImportError as e:
    print("Error: Required packages not found. Please install them using:")
    print("pip install -r requirements.txt")
    print("\nDetailed error:", str(e))
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python onnx_any_demo.py <path_to_onnx_model>")
        print("\nExample:")
        print("  python onnx_any_demo.py models/resnet50.onnx")
        sys.exit(1)

    onnx_path = Path(sys.argv[1])
    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        sys.exit(1)

    # Create temp directory for mount point
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mount_point = tmpdir / "mount"
        mount_point.mkdir()
        
        # Launch MLFS
        mlfs_py = os.path.join(os.path.dirname(__file__), '..', 'mlfs.py')
        proc = subprocess.Popen(
            [sys.executable, mlfs_py, '--model', str(onnx_path), '--mount', str(mount_point), '--foreground'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        # Show first log line
        print("│ mlfs │", proc.stdout.readline().rstrip())
        
        # Wait until /model shows up (max 3 s)
        for _ in range(30):
            if os.path.isdir(os.path.join(mount_point, 'model')):
                break
            time.sleep(0.1)
        else:
            print("❌  mount still empty after 3 s – bailing out.")
            proc.terminate()
            shutil.rmtree(mount_point)
            sys.exit(1)
        
        try:
            # Show model info
            print("\nModel info:")
            subprocess.run(['cat', os.path.join(mount_point, 'sys', 'model_str')])
            
            # Explore the mounted filesystem
            print("\n== directory tree ==")
            subprocess.run(['find', str(mount_point), '-maxdepth', '3', '-print'])
            
            # List all weight tensors
            print("\nAvailable weight tensors:")
            subprocess.run(['find', str(mount_point), '-name', '*.bin', '-type', 'f'])
            
            # Show first tensor's contents (if any)
            weight_files = list(Path(mount_point).rglob('*.bin'))
            if weight_files:
                first_weight = weight_files[0]
                print(f"\nContents of {first_weight.name}:")
                subprocess.run(['hexdump', '-C', str(first_weight), '-n', '16'])
            
            input("\nPress Enter to unmount...")
            
        finally:
            # Cleanup
            print("\n🛑  Unmounting …")
            subprocess.run(['fusermount', '-u', str(mount_point)])
            proc.terminate()
            print("✅  Done.")

if __name__ == "__main__":
    main() 