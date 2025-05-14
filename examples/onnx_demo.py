#!/usr/bin/env python3
"""
ONNX Demo for MLFS
Creates a simple ONNX model and mounts it using MLFS
"""

import os
import sys
import tempfile
import subprocess
import shutil
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import onnx
    from onnx import version_converter
except ImportError as e:
    print("Error: Required packages not found. Please install them using:")
    print("pip install -r requirements.txt")
    print("\nDetailed error:", str(e))
    sys.exit(1)

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def main():
    # Create temp directory for our demo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and export ONNX model
        model = SimpleModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 10)
        
        # Export to ONNX
        onnx_path = tmpdir / "simple_model.onnx"
        try:
            # Export with opset version 12 for better compatibility
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}},
                opset_version=12,  # Use a stable opset version
                do_constant_folding=True
            )
            
            # Verify the ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
        except Exception as e:
            print("Error exporting model to ONNX:", str(e))
            print("\nTroubleshooting tips:")
            print("1. Make sure you have the correct versions installed:")
            print("   pip install onnx==1.15.0 protobuf==3.20.3")
            print("2. If using a virtual environment, make sure it's activated")
            print("3. Try removing and reinstalling onnx and protobuf:")
            print("   pip uninstall onnx protobuf")
            print("   pip install onnx==1.15.0 protobuf==3.20.3")
            sys.exit(1)
        
        # Create mount point
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
            # Explore the mounted filesystem
            print("\n== directory tree ==")
            subprocess.run(['find', str(mount_point), '-maxdepth', '3', '-print'])
            
            # Show some tensor contents
            print("\nContents of linear1.weight:")
            subprocess.run(['hexdump', '-C', os.path.join(mount_point, 'model', 'linear1.weight.bin'), '-n', '16'])
            
            # Show model info
            print("\nModel info:")
            subprocess.run(['cat', os.path.join(mount_point, 'sys', 'model_str')])
            
            input("\nPress Enter to unmount...")
            
        finally:
            # Cleanup
            print("\n🛑  Unmounting …")
            subprocess.run(['fusermount', '-u', str(mount_point)])
            proc.terminate()
            print("✅  Done.")

if __name__ == "__main__":
    main() 