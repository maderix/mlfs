#!/usr/bin/env python3
"""
MLFS – Machine‑Learning FileSystem

examples
--------
# safe (weights read‑only, grads writable)
python mlfs.py --model resnet18.pt --mount /mnt/mlfs --foreground --wait

# allow raw weight edits (dangerous – trust your code)
python mlfs.py --model resnet18.pt --mount /mnt/mlfs --foreground --wait --unsafe-write
"""
import os, sys, errno, time, stat, argparse, logging
from collections import OrderedDict
from threading import RLock

import torch
from fuse import FUSE, FuseOSError, Operations       # pip install fusepy

# ────────── helpers ──────────────────────────────────────────────────────────
def _now() -> int: return int(time.time())

class VNode:
    """minimal in‑memory inode."""
    def __init__(self, name: str, mode: int):
        self.name = name
        self.mode = mode
        self.atime = self.mtime = self.ctime = _now()
        self.children: "OrderedDict[str, VNode]" = OrderedDict()
        self.data: bytes = b""

    def stat(self):
        return dict(
            st_mode  = self.mode,
            st_nlink = 2 if stat.S_ISDIR(self.mode) else 1,
            st_size  = len(self.data),
            st_ctime = self.ctime,
            st_mtime = self.mtime,
            st_atime = self.atime,
        )

# ────────── build FS tree from torch model ───────────────────────────────────
def build_tree(model) -> VNode:
    root = VNode("/", stat.S_IFDIR | 0o755)

    # /sys
    sysd = VNode("sys", stat.S_IFDIR | 0o555); root.children["sys"] = sysd
    for k, v in {"version": "0.3", "model_str": str(model)}.items():
        f = VNode(k, stat.S_IFREG | 0o444); f.data = v.encode(); sysd.children[k] = f

    # /model
    modeld = VNode("model", stat.S_IFDIR | 0o755); root.children["model"] = modeld
    for name, p in model.named_parameters():
        parent = modeld
        for part in name.split(".")[:-1]:
            parent = parent.children.setdefault(part, VNode(part, stat.S_IFDIR | 0o755))

        w = VNode(name.split(".")[-1] + ".bin", stat.S_IFREG | 0o444)
        w.data = p.detach().cpu().numpy().tobytes(); parent.children[w.name] = w

        g = VNode(name.split(".")[-1] + ".grad", stat.S_IFREG | 0o666)
        g.data = b'' if p.grad is None else p.grad.cpu().numpy().tobytes()
        parent.children[g.name] = g

    # /activations placeholder
    root.children["activations"] = VNode("activations", stat.S_IFDIR | 0o755)

    # /ir
    ird = VNode("ir", stat.S_IFDIR | 0o444); root.children["ir"] = ird
    graph_txt = torch.jit.script(model).graph.__str__().encode()
    txt = VNode("graph.txt", stat.S_IFREG | 0o444); txt.data = graph_txt; ird.children["graph.txt"] = txt

    # /logs
    logd = VNode("logs", stat.S_IFDIR | 0o444); root.children["logs"] = logd
    logd.children["fuse.log"] = VNode("fuse.log", stat.S_IFREG | 0o444)

    return root

# ────────── FUSE implementation ──────────────────────────────────────────────
class MLFS(Operations):
    def __init__(self, model_path: str, allow_bin_write: bool = False):
        self.rwlock = RLock()
        self.allow_bin_write = allow_bin_write
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.root  = build_tree(self.model)
        logging.info("Loaded model %s (allow_bin_write=%s)", model_path, allow_bin_write)

    # path resolver
    def _resolve(self, path: str) -> VNode:
        node = self.root
        for part in filter(None, path.strip("/").split("/")):
            if not stat.S_ISDIR(node.mode):
                raise FuseOSError(errno.ENOTDIR)
            node = node.children.get(part)
            if node is None:
                raise FuseOSError(errno.ENOENT)
        return node

    # FUSE callbacks
    def getattr(self, path, fh=None):
        with self.rwlock:
            return self._resolve(path).stat()

    def readdir(self, path, fh):
        with self.rwlock:
            node = self._resolve(path)
            if not stat.S_ISDIR(node.mode):
                raise FuseOSError(errno.ENOTDIR)
            return [".", ".."] + list(node.children)

    def read(self, path, size, offset, fh):
        with self.rwlock:
            data = self._resolve(path).data
            return data[offset: offset + size]

    def _is_weight_bin(self, path: str) -> bool:
        return path.startswith("/model") and path.endswith(".bin")

    def _is_grad(self, path: str) -> bool:
        return path.startswith("/model") and path.endswith(".grad")

    # write handler
    def write(self, path, buf, offset, fh):
        with self.rwlock:
            node = self._resolve(path)

            if self._is_grad(path) or (self.allow_bin_write and self._is_weight_bin(path)):
                node.data = node.data[:offset] + buf
                return len(buf)

            if path == "/logs/fuse.log":
                node.data += buf
                return len(buf)

            raise FuseOSError(errno.EPERM)

    # honor O_TRUNC in open()
    def open(self, path, flags):
        if flags & os.O_TRUNC:
            self.truncate(path, 0)
        return 0

    # explicit truncate
    def truncate(self, path, length, fh=None):
        with self.rwlock:
            node = self._resolve(path)
            if length == 0 and (self._is_grad(path) or (self.allow_bin_write and self._is_weight_bin(path))):
                node.data = b""
                return 0
            raise FuseOSError(errno.EPERM)

    def flush  (self, path, fh): return 0
    def release(self, path, fh): return 0

# ────────── CLI ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MLFS – mount a model as a filesystem")
    ap.add_argument("--model", required=True, help="model checkpoint (.pt/.pth)")
    ap.add_argument("--mount", required=True, help="existing directory to mount on")
    ap.add_argument("-f", "--foreground", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="exit only after /model is visible (for script callers)")
    ap.add_argument("--unsafe-write", action="store_true",
                    help="ALLOW writes to *.bin weight files (dangerous)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[mlfs] %(message)s")
    if args.foreground and args.wait:
        logging.warning("--wait ignored in foreground mode (process blocks).")
        args.wait = False

    FUSE(
        MLFS(args.model, allow_bin_write=args.unsafe_write),
        args.mount,
        foreground=args.foreground,
        ro=False,
        allow_other=True,
    )

    # simple readiness poll for scripts
    if args.wait:
        for _ in range(60):                 # up to ~6 s
            if os.path.isdir(os.path.join(args.mount, "model")):
                break
            time.sleep(0.1)

if __name__ == "__main__":
    main()
