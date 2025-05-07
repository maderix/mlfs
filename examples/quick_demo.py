#!/usr/bin/env python3
"""
quick_demo.py – MLFS smoke‑test with a toy nn.Sequential.

Run from repo root:
    source venv/bin/activate
    python examples/quick_demo.py
"""
import os, sys, subprocess, tempfile, shutil, time, torch, torch.nn as nn

# -- 1. build tiny model ------------------------------------------------------
model_path = 'tiny.pt'
torch.save(nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)), model_path)

# -- 2. temp mountpoint -------------------------------------------------------
mnt = tempfile.mkdtemp(prefix='mlfs_')
print(f"🎬  Mounting MLFS at {mnt}")

# -- 3. launch MLFS -----------------------------------------------------------
mlfs_py = os.path.join(os.path.dirname(__file__), '..', 'mlfs.py')
proc = subprocess.Popen(
    [sys.executable, mlfs_py, '--model', model_path, '--mount', mnt, '--foreground'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# show first log line
print("│ mlfs │", proc.stdout.readline().rstrip())

# wait until /model shows up (max 3 s)
for _ in range(30):
    if os.path.isdir(os.path.join(mnt, 'model')):
        break
    time.sleep(0.1)
else:
    print("❌  mount still empty after 3 s – bailing out.")
    proc.terminate(); shutil.rmtree(mnt); sys.exit(1)

# -- 4. interact -------------------------------------------------------------
print("\n== directory tree ==")
subprocess.run(['find', mnt, '-maxdepth', '3', '-print'])

wfile = os.path.join(mnt, 'model', '0', 'weight.bin')
print("\nFirst 16 bytes of first weight matrix:")
subprocess.run(['hexdump', '-C', wfile, '-n', '16'])

gfile = os.path.join(mnt, 'model', '0', 'weight.grad')
print("\n✏️  Writing fake gradient byte …")
with open(gfile, 'r+b') as f:
    f.write(b'\x42')

print("Gradient first byte now:")
subprocess.run(['hexdump', '-C', gfile, '-n', '1'])

# -- 5. teardown -------------------------------------------------------------
print("\n🛑  Unmounting …")
subprocess.run(['fusermount', '-u', mnt])
proc.terminate()
shutil.rmtree(mnt)
os.remove(model_path)
print("✅  Done.")
