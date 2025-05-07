# 🍣 MLFS – Machine Learning File­system 🍣

*Mount your model like a USB stick and `cd` straight into its brain.*

---

## 🤔  What‐the‐heck‑is‑this?

MLFS is a tiny <abbr title="Filesystem in Userspace">FUSE</abbr> driver that turns a **PyTorch** model into a directory tree:

```
/sys/          ← version, banner, full model string
/model/        ← every layer → folder  •   tensors → files  •   grads → *.grad
/activations/  ← forward‑pass dumps
/ir/           ← TorchScript graph
/logs/         ← tail ‑f me plz
```

`ls`, `cat`, `diff`, `grep`… if your shell can touch it, it can now poke your network’s guts.

---

## ⚡️ 10‑second install

```bash
sudo apt install fuse              # (Linux) – macOS: brew install macfuse
pip install torch torchvision fusepy
```

---

## 🎸  Mount n’ Roll

```bash
# make a playground mountpoint
sudo mkdir -p /mnt/mlfs

# export any .pt you like (here we cook a ResNet‑18)
python - <<'PY'
import torch, torchvision as tv
torch.save(tv.models.resnet18(weights=None), 'resnet18.pt')
PY

# fire it up (foreground, ctrl‑c to quit)
sudo ./mlfs.py --model resnet18.pt --mount /mnt/mlfs --foreground
```

Now try the shell‑fu:

```bash
ls /mnt/mlfs/model/conv1
hexdump -C /mnt/mlfs/model/conv1/weight.bin | head -n 4

echo "hello log👋" | sudo tee -a /mnt/mlfs/logs/fuse.log
```

Unmount when done:

```bash
sudo fusermount -u /mnt/mlfs   # linux
# sudo umount /mnt/mlfs        # macOS
```

---

## 🐣  Example 1 — Tiny MLP *quick\_demo.py*

```bash
cd mlfs/examples
sudo python3 quick_demo.py
```

Watch it:

1. Builds a 2‑layer MLP on the fly.
2. Mounts MLFS in a temp dir.
3. Hex‑spills the first 16 bytes of the weight matrix.
4. Patches one gradient byte (because we can ✨).
5. Cleans all traces like a ninja.

---

## 🦖  Example 2 — ResNet‑18 *resnet\_demo.py*

```bash
sudo python3 examples/resnet_demo.py
```

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

## 🗺️  Roadmap (a.k.a. TODO or bust)

* 🐛 Refactor single‑file prototype → proper package.
* 📦 `pip install mlfs` dream.
* 📊 Auto‑populate `/activations/<timestamp>/` on forward hooks.
* ✍️ Writable weight surgery (`echo 0.0 > …/weight.bin`).
* 🌌 mmap‑backed giant‑model support.

---

## 🧞‍♂️  License

MIT.  Go forth and mount exotically.

> “**Everything is a file**… even a 175‑billion‑parameter fever dream.” – some POSIX wizard
