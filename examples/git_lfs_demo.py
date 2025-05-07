#!/usr/bin/env python3
"""
git_time_machine_demo.py
------------------------
Commit, diff, and roll back Inception v3 weights with ordinary Git.

* Saves Inception v3 **state‑dict** (100 MB, faster than full Module)
* Mounts MLFS with `--state-dict --unsafe-write`
* Commits /model as **v1** (uses Git‑LFS if available)
* Flips 1 byte of the first weight file via MLFS, commits **v2**
* Shows 1‑byte diff, `git checkout HEAD~1` rolls weights back instantly
"""

import os, sys, subprocess, tempfile, shutil, json, time
import torch, torchvision as tv

# ---------- tiny helpers -----------------------------------------------------
def run(cmd, **kw): subprocess.run(cmd, check=True, **kw)

def have_git_lfs() -> bool:
    return subprocess.run(
        ["git", "lfs", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode == 0

# ---------- scratch repo -----------------------------------------------------
repo = tempfile.mkdtemp(prefix="mlfs_git_demo_")
os.chdir(repo)
run(["git", "init", "-q"])

if have_git_lfs():
    run(["git", "lfs", "install", "--local"])
    print("✅ git‑lfs enabled")
else:
    print("⚠️  git‑lfs not found; committing raw blobs")

# ---------- save Inception state‑dict ---------------------------------------
ckpt = "inception_v3_sd.pt"
if not os.path.exists(ckpt):
    print("🔻 downloading Inception v3 weights …")
    sd = tv.models.inception_v3(
        weights=tv.models.Inception_V3_Weights.IMAGENET1K_V1
    ).state_dict()
    torch.save(sd, ckpt)

# ---------- mount MLFS -------------------------------------------------------
mnt = tempfile.mkdtemp(prefix="mlfs_")
mlfs_py = os.path.join(os.path.dirname(__file__), "..", "mlfs.py")
mlfs = subprocess.Popen(
    [
        sys.executable, mlfs_py,
        "--model", ckpt,
        "--state-dict",
        "--unsafe-write",
        "--mount", mnt,
        "--wait"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT
)

root_model = os.path.join(mnt, "model")

# locate the first weight.bin
wfile = None
for _ in range(120):                      # 12 s max
    for r, _, files in os.walk(root_model):
        if "weight.bin" in files:
            wfile = os.path.join(r, "weight.bin")
            break
    if wfile:
        break
    time.sleep(0.1)
if wfile is None:
    sys.exit("weight.bin not found – aborting")

# ---------- commit v1 --------------------------------------------------------
run(["cp", "-r", root_model, "."])
run(["git", "add", "model"])
run(["git", "commit", "-m", "v1", "-q"])
print("✅ committed v1")

# ---------- flip one byte & commit v2 ----------------------------------------
with open(wfile, "r+b") as f:
    buf = bytearray(f.read())
    buf[0] ^= 0x80
    f.seek(0); f.write(buf)

shutil.rmtree("model", ignore_errors=True)     # remove stale tree
run(["cp", "-r", root_model, "."])
run(["git", "add", "model"])
run(["git", "commit", "-m", "v2", "-q"])
print("✅ committed v2 (1‑byte change)")

# ---------- diff & rollback --------------------------------------------------
print("\n=== git diff v1..v2 ===")
rel_path = os.path.relpath(wfile, root_model)
run(["git", "diff", "--color", "--", f"model/{rel_path}"])

print("\nRolling back to v1 …")
run(["git", "checkout", "-q", "HEAD~1"])
byte0 = open(f"model/{rel_path}", "rb").read(1)[0]
print(json.dumps({"first_byte_after_checkout": byte0}, indent=2))

# ---------- cleanup ----------------------------------------------------------
run(["fusermount", "-u", mnt])
mlfs.terminate()
shutil.rmtree(mnt)
print(f"\nRepo left at {repo} — explore with Git commands if you like.")
