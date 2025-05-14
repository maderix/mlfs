# 💃🔥 MLFS – Machine Learning Filesystem 💃🔥

*Mount your model like a USB stick and `cd` straight into its brain.*

---

## 🤔  What‐the‐heck‑is‑this?

MLFS is a tiny <abbr title="Filesystem in Userspace">FUSE</abbr> driver that turns a **PyTorch** or **ONNX** model into a directory tree:

```
/sys/          ← version, banner, full model string
/model/        ← every layer → folder  •   tensors → files  •   grads → *.grad
/activations/  ← forward‑pass dumps
/ir/           ← TorchScript/ONNX graph
/logs/         ← tail ‑f me plz
```

`ls`, `cat`, `diff`, `grep`… if your shell can touch it, it can now poke your network's guts.

---

## ⚡️ 10‑second install

```bash
# Linux
apt install fuse              # or your distro's package manager
# macOS
brew install macfuse

# Python packages
pip install torch torchvision fusepy onnx
```

---

## 🎸  Mount n' Roll

```bash
# Create a temporary mount point in your home directory
mkdir -p ~/mlfs_mount

# Export any .pt or .onnx you like (here we cook a ResNet‑18)
python - <<'PY'
import torch, torchvision as tv
torch.save(tv.models.resnet18(weights=None), 'resnet18.pt')
PY

# Fire it up (foreground, ctrl‑c to quit)
./mlfs.py --model resnet18.pt --mount ~/mlfs_mount --foreground
```

Now try the shell‑fu:

```bash
ls ~/mlfs_mount/model/conv1
hexdump -C ~/mlfs_mount/model/conv1/weight.bin | head -n 4

echo "hello log👋" >> ~/mlfs_mount/logs/fuse.log
```

Unmount when done:

```bash
fusermount -u ~/mlfs_mount   # linux
# umount ~/mlfs_mount        # macOS
```

---

## 🐣  Example 1 — Tiny MLP *quick_demo.py*

```bash
cd mlfs/examples
python3 quick_demo.py
```

Watch it:

1. Builds a 2‑layer MLP on the fly.
2. Mounts MLFS in a temp dir.
3. Hex‑spills the first 16 bytes of the weight matrix.
4. Patches one gradient byte (because we can ✨).
5. Cleans all traces like a ninja.

---

## 🦖  Example 2 — ResNet‑18 *resnet_demo.py*

```bash
python3 examples/resnet_demo.py
```

## 🕰️ Git‑Time‑Machine Demo (🚀 new!)
```bash
python examples/git_time_machine_demo.py
```
🕸️ MLFS mounts Inception v3 as regular files.
```
📸 git commit of /model = v1.
```
🔪 Flip one byte in a weight via the FS, commit again = v2.
```
git diff shows a ☝️‑byte hex delta.
```
⏪ git checkout v1 rewinds the network instantly—no reloads, no downtime.

## 🎉 Your neural net now responds to git log, git diff, and git checkout like any ordinary code repo.  Time‑travel debugging with zero custom tools!

## 🦊 Example 3 — ONNX Model *onnx_demo.py* (🚀 new!)
```bash
python3 examples/onnx_demo.py
```
This demo:
1. Creates a simple neural network
2. Exports it to ONNX format
3. Mounts it using MLFS
4. Shows how to explore the model structure and weights

This heftier demo lists deep sub‑blocks (`layer1/0/conv1`) and dumps the raw conv1 tensor to prove MLFS handles real architectures.

---

## 🛠️ 12 actually‑useful things you can do (a totally serious list)

| #  | Use‑case                                                                   | Emoji vibe |
| -- | -------------------------------------------------------------------------- | ---------- |
| 1  | **Unix‑native surgery** – `vim` a bias, save, watch trainer hot‑reload     | 🩺         |
| 2  | **Activation scooping** – `inotifywait` + CSV dumps, zero extra hooks      | 🍦         |
| 3  | **Time‑travel debugging** – `git checkout step‑1337` and remount           | ⌛️         |
| 4  | **Quick viz** – point TensorBoard at weight files (no writer boiler‑plate) | 🖼️        |
| 5  | **Remote prod inspection** – `sshfs` into a containerized model            | 🛰️        |
| 6  | **CI sanity** – fail build if any `*.grad` == 0 bytes 🤨                   | 🔨         |
| 7  | **Observability** – Grafana agent tails `/logs/fuse.log`                   | 📉         |
| 8  | **Fuse‑RPC inference** – write input tensor, read output, no gRPC          | 📡         |
| 9  | **Edutainment** – students literally `cd` into ResNet tunnels              | 🎓         |
| 10 | **Security diff** – hash all blobs, alarm on tamper                        | 🛡️        |
| 11 | **Delta checkpoints** – `rsync --inplace` on file blobs                    | 💾         |
| 12 | **Lazy mmap serving** – roadmap: weights stay on disk, page‑in live        | 🛸         |

---


## 🔭 Roadmap — where MLFS can venture next 🚀

| Track | What it buys you | Effort estimate | Caveats / Gotchas |
|-------|------------------|-----------------|-------------------|
| **ONNX / TensorFlow / TFLite back‑ends** | Mount *any* framework's weights→ same tree → cross‑tool diffing & hacks | **Medium** – parse each format once, expose tensors as `memoryview` | ONNX & TF easy (protobuf blobs); TFLite flatbuffers parser needed |
| **GPU‑mmap weights (GPUDirect Storage)** | Model bytes stream straight from file into GPU memory – no CPU copy, instant warm‑up | **Hard** – integrate `cuFile` / `cudaMallocHost` pinned pages | Requires A100/H100‑class HW + kernel mods |
| **Lazy safetensors index** | Sub‑second mount of 20 GB Llama; chunk‑SHA dedup across checkpoints | **Medium** – we already mmap safetensors, add index & cache | — |
| **Write‑back plugin** | FS edits → regenerate `.pt` / `.safetensors` automatically | **Easy** – invert `build_tree`, embed dtype/shape in `/sys` | — |
| **Activation shared‑memory taps** | Another process can `mmap` live feature maps → real‑time viz | **Medium‑Hard** – expose `/activations/*` as shm, need sync | Needs inotify/futex protocol |
| **FUSE‑dmabuf zero‑copy tensors** | Kernel hands GPU a dmabuf handle → true zero‑copy train | **Research** – pending upstream FUSE patches | bleeding‑edge kernel only |

> **Legend** – *Effort* ≈ weekend hack = Easy, few weekends = Medium, research rabbit‑hole = Hard


---

## 🧞‍♂️  License

MIT.  Go forth and mount exotically.

> "**Everything is a file**… even a 175‑billion‑parameter fever dream." – some POSIX wizard
