#!/usr/bin/env python3
"""
conv1_surgery_demo.py – ‘hot‑knife’ model hacking via MLFS (safe edition)

Steps
-----
1. Save an untrained torchvision ResNet‑18 checkpoint.
2. Mount MLFS (same venv interpreter, --wait for readiness).
3. Run a dummy forward pass → baseline logits.
4. Read conv1/weight.bin, flip the sign of its first 16 floats,
   then write the hacked bytes to conv1/weight.grad (writable FS node).
5. Copy those hacked bytes into the live tensor → logits mutate.
6. Unmount and clean up.

Run from repo root (venv active):

    python examples/conv1_surgery_demo.py

If your user still needs root for FUSE, prefix with
    sudo -E $(which python) examples/conv1_surgery_demo.py
"""

import os, sys, subprocess, tempfile, shutil, time
import torch, torchvision as tv
import numpy as np

torch.manual_seed(0)

# 0 · checkpoint --------------------------------------------------------------
ckpt = "resnet18.pt"
torch.save(tv.models.resnet18(weights=None), ckpt)

# 1 · mount MLFS --------------------------------------------------------------
mnt = tempfile.mkdtemp(prefix="mlfs_")
mlfs_py = os.path.join(os.path.dirname(__file__), "..", "mlfs.py")
mlfs_proc = subprocess.Popen(
    [sys.executable, mlfs_py, "--model", ckpt, "--mount", mnt, "--wait"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)

# wait until conv1 weight appears
wbin = os.path.join(mnt, "model", "conv1", "weight.bin")
wgrad = os.path.join(mnt, "model", "conv1", "weight.grad")
for _ in range(40):          # ~4 s
    if os.path.exists(wbin) and os.path.exists(wgrad):
        break
    time.sleep(0.1)
else:
    print("❌ conv1 files not found – aborting.")
    mlfs_proc.terminate(); shutil.rmtree(mnt); sys.exit(1)

# 2 · baseline inference ------------------------------------------------------
model = torch.load(ckpt, weights_only=False)
model.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    baseline = model(x)

print("baseline logits   →", baseline[0][:5])

# 3 · read + hack conv1 weights ----------------------------------------------
with open(wbin, "rb") as f:
    buf = bytearray(f.read())

floats = np.frombuffer(buf, dtype="<f4")
print("first 4 floats pre‑op:", floats[:4])

floats[:16] *= -1.0                # flip sign of first 16 floats
hacked_bytes = floats.tobytes()

# dump hacked bytes into weight.grad (writable)
with open(wgrad, "r+b") as f:
    f.write(hacked_bytes)
print("🩺 dumped hacked bytes into weight.grad")

# 4 · reflect hack into live tensor ------------------------------------------
patch = torch.from_numpy(
    floats[: model.conv1.weight.numel()]
).view_as(model.conv1.weight)

with torch.no_grad():
    model.conv1.weight.copy_(patch)

# 5 · post‑surgery inference --------------------------------------------------
with torch.no_grad():
    hacked = model(x)

print("post‑surgery logits→", hacked[0][:5])
print("L2 diff            →", (baseline - hacked).pow(2).sum().sqrt().item())

# 6 · cleanup -----------------------------------------------------------------
print("\n🛑  Unmounting …")
subprocess.run(["fusermount", "-u", mnt], check=True)
mlfs_proc.terminate()
shutil.rmtree(mnt)
os.remove(ckpt)
print("✅  Done.")
