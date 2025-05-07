#!/usr/bin/env python3
"""
grad_sabotage_demo.py – show gradient truncation via MLFS.
"""
import os, sys, subprocess, tempfile, shutil, time, torch, torch.nn as nn

torch.manual_seed(0)

# 1 · build tiny model --------------------------------------------------------
model = nn.Linear(4, 1, bias=False)
orig_w = model.weight.data.clone()
ckpt   = "linear.pt";  torch.save(model, ckpt)

# 2 · mount MLFS --------------------------------------------------------------
mnt = tempfile.mkdtemp(prefix="mlfs_")
mlfs_py = os.path.join(os.path.dirname(__file__), "..", "mlfs.py")
mlfs = subprocess.Popen(
    [sys.executable, mlfs_py, "--model", ckpt, "--mount", mnt, "--wait"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)

grad_file = os.path.join(mnt, "model", "weight.grad")

# wait until grad file appears (should be instant)
for _ in range(30):             # 3 s max
    if os.path.exists(grad_file):
        break
    time.sleep(0.1)

# 3 · forward + backward ------------------------------------------------------
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(8, 4), torch.randn(8, 1)
loss = ((model(x) - y) ** 2).mean(); loss.backward()

print("🔎 grad tensor pre‑sabotage:", model.weight.grad.flatten())
print("   grad file size          :", os.path.getsize(grad_file), "bytes")

# 4 · sabotage via truncate ---------------------------------------------------
with open(grad_file, "wb"):      # O_TRUNC handled by mlfs.py
    pass
for p in model.parameters():
    p.grad.zero_()

print("💥 grad tensor after zero   :", model.weight.grad.flatten())
print("   grad file size now       :", os.path.getsize(grad_file), "bytes")

# 5 · optimizer step (should do nothing) -------------------------------------
opt.step()

print("\nInitial weight:\n", orig_w)
print("Current weight:\n", model.weight.data)
print("Sabotage worked ✅" if torch.allclose(orig_w, model.weight.data) else "FAILED ❌")

# 6 · cleanup -----------------------------------------------------------------
subprocess.run(["fusermount", "-u", mnt])
mlfs.terminate()
shutil.rmtree(mnt); os.remove(ckpt)
