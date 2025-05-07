#!/usr/bin/env python3
"""
resnet_demo.py – mount ResNet‑18 in MLFS, peek at conv1 weights.

Run:
    source venv/bin/activate
    python examples/resnet_demo.py
If your user can’t access FUSE, run the whole command with sudo -E.
"""
import os, sys, subprocess, tempfile, shutil, time, torch, torchvision as tv

# 1 · save checkpoint ---------------------------------------------------------
ckpt = "resnet18.pt"
torch.save(tv.models.resnet18(weights=None), ckpt)

# 2 · temp mount --------------------------------------------------------------
mnt = tempfile.mkdtemp(prefix="mlfs_")
print(f"📦  Mounting MLFS at {mnt}")

mlfs_py = os.path.join(os.path.dirname(__file__), "..", "mlfs.py")
proc = subprocess.Popen(
    [sys.executable, mlfs_py, "--model", ckpt, "--mount", mnt, "--wait"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

print("│ mlfs │", proc.stdout.readline().rstrip())

# 3 · wait for deep path to appear -------------------------------------------
wfile = os.path.join(mnt, "model", "layer1", "0", "conv1", "weight.bin")
for _ in range(40):                # ~4 s max
    if os.path.exists(wfile):
        break
    time.sleep(0.1)
else:
    print("❌  conv1 not yet visible – aborting.")
    proc.terminate(); shutil.rmtree(mnt); sys.exit(1)

# 4 · show directory + hexdump -----------------------------------------------
print("\n== /model/layer1/0/conv1 ==")
subprocess.run(["find", os.path.dirname(wfile), "-maxdepth", "1", "-print"])

print("\nFirst 64 bytes of conv1 weight tensor:")
subprocess.run(["hexdump", "-C", wfile, "-n", "64"], check=True)

# 5 · clean up ---------------------------------------------------------------
print("\n🛑  Unmounting …")
subprocess.run(["fusermount", "-u", mnt], check=True)
proc.terminate()
shutil.rmtree(mnt)
os.remove(ckpt)
print("✅  Done.")
