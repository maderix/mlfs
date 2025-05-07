#!/usr/bin/env python3
"""
mmap_demo.py – zero‑copy peek into a large transformer weight via MLFS.

1. Download + save Hugging‑Face GPT‑2‑medium *state‑dict* (~1.5 GB) once.
2. Mount MLFS with --state-dict; wait for weight.bin to appear.
3. mmap that file, print first few floats + mean/std without loading to RAM.
4. Show peak RSS (<30 MB), unmount, done.
"""

import os, sys, mmap, struct, subprocess, tempfile, shutil, time, psutil
import torch
from transformers import GPT2LMHeadModel

MODEL_NAME = "gpt2-medium"
CKPT_FILE  = "gpt2_medium_sd.pt"     # state‑dict file

# ── 1. checkpoint download / cache ──────────────────────────────────────────
if not os.path.exists(CKPT_FILE):
    print("🔻 downloading GPT‑2‑medium (first run) …")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    torch.save(model.state_dict(), CKPT_FILE)        # state‑dict only
else:
    print("⏩ using cached state‑dict")

# ── 2. mount MLFS (state‑dict mode) ─────────────────────────────────────────
mnt = tempfile.mkdtemp(prefix="mlfs_")
mlfs_py = os.path.join(os.path.dirname(__file__), "..", "mlfs.py")
mlfs = subprocess.Popen(
    [
        sys.executable, mlfs_py,
        "--model", CKPT_FILE,
        "--state-dict",
        "--mount", mnt,
        "--wait",
    ],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)

wpath = os.path.join(
    mnt, "model", "transformer", "h", "0", "attn", "c_attn", "weight.bin"
)

print("⏳ waiting for MLFS to expose q‑proj weight …", end="", flush=True)
for i in range(240):                # up to 4 min (should appear in ~20 s on SSD)
    if os.path.exists(wpath):
        break
    time.sleep(0.5)
    if i % 8 == 0:                  # dot every 4 s
        print(".", end="", flush=True)
else:
    sys.exit("\nweight file not found – aborting")
print(" found!")

# ── 3. mmap + stats ─────────────────────────────────────────────────────────
with open(wpath, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    first_vals = struct.unpack("<8f", mm[:32])
    print("first 8 floats:", first_vals)

    total = len(mm) // 4
    s = ss = 0.0
    stride = 1_000_000           # 4 MB
    for off in range(0, len(mm), stride * 4):
        cnt = min(stride, total - off // 4)
        chunk = struct.unpack("<" + "f"*cnt, mm[off : off + cnt*4])
        s  += sum(chunk)
        ss += sum(x*x for x in chunk)
    mean = s / total
    std  = (ss / total - mean*mean) ** 0.5
    print(f"mean={mean:.5f}  std={std:.5f}")

rss = psutil.Process().memory_info().rss / (1024*1024)
print(f"Peak RSS of this script: {rss:.1f} MB")

# ── 4. cleanup ──────────────────────────────────────────────────────────────
print("🛑 unmounting …")
subprocess.run(["fusermount", "-u", mnt])
mlfs.terminate()
shutil.rmtree(mnt)
print("✅ demo complete")
