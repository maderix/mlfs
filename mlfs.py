#!/usr/bin/env python3
"""
MLFS – Machine‑Learning FileSystem  (v 0.6)

* Zero‑copy memoryview weights  → fast mount, low RAM
* --state-dict loads a raw weights file (no nn.Module unpickle)
* --unsafe-write still lets you edit *.bin if you dare
* ONNX model support added
"""

import os, sys, errno, time, stat, argparse, logging
from collections import OrderedDict
from threading import RLock
from typing import Mapping, Union
import numpy as np

import torch
import onnx
from fuse import FUSE, FuseOSError, Operations

# ───────── helpers ──────────────────────────────────────────────────────────
def _now() -> int: return int(time.time())

class VNode:
    def __init__(self, name: str, mode: int):
        self.name = name
        self.mode = mode
        self.atime = self.mtime = self.ctime = _now()
        self.children: "OrderedDict[str,VNode]" = OrderedDict()
        self.data: memoryview | bytes = b""

    def stat(self):
        return dict(
            st_mode  = self.mode,
            st_nlink = 2 if stat.S_ISDIR(self.mode) else 1,
            st_size  = len(self.data),
            st_ctime = self.ctime,
            st_mtime = self.mtime,
            st_atime = self.atime,
        )

# ───────── build tree ───────────────────────────────────────────────────────
def build_tree(source: Union[Mapping[str, torch.Tensor], onnx.ModelProto], meta_str: str) -> VNode:
    root = VNode("/", stat.S_IFDIR | 0o755)

    sysd = VNode("sys", stat.S_IFDIR | 0o555); root.children["sys"] = sysd
    for k, v in {"version": "0.6‑mview", "model_str": meta_str}.items():
        f = VNode(k, stat.S_IFREG | 0o444); f.data = v.encode(); sysd.children[k] = f

    modeld = VNode("model", stat.S_IFDIR | 0o755); root.children["model"] = modeld

    if isinstance(source, onnx.ModelProto):
        # Handle ONNX model
        for initializer in source.graph.initializer:
            name = initializer.name
            
            # Get the correct dtype from the tensor
            dtype_map = {
                1: np.float32,
                2: np.uint8,
                3: np.int8,
                4: np.uint16,
                5: np.int16,
                6: np.int32,
                7: np.int64,
                8: np.bytes_,
                9: np.bool_,
                10: np.float16,
                11: np.float64,
                12: np.uint32,
                13: np.uint64,
                14: np.complex64,
                15: np.complex128,
            }
            dtype = dtype_map.get(initializer.data_type, np.float32)
            
            # Handle different data storage formats
            if initializer.raw_data:
                tensor = np.frombuffer(initializer.raw_data, dtype=dtype)
            elif initializer.float_data:
                tensor = np.array(initializer.float_data, dtype=np.float32)
            elif initializer.int32_data:
                tensor = np.array(initializer.int32_data, dtype=np.int32)
            elif initializer.int64_data:
                tensor = np.array(initializer.int64_data, dtype=np.int64)
            elif initializer.string_data:
                tensor = np.array(initializer.string_data, dtype=np.bytes_)
            else:
                logging.warning(f"Skipping tensor {name} - no data found")
                continue
                
            # Reshape if dimensions are provided
            if initializer.dims:
                try:
                    tensor = tensor.reshape(initializer.dims)
                except ValueError as e:
                    logging.warning(f"Could not reshape tensor {name}: {e}")
                    continue
            
            parent = modeld
            for part in name.split("/")[:-1]:
                parent = parent.children.setdefault(part, VNode(part, stat.S_IFDIR | 0o755))

            w = VNode(name.split("/")[-1] + ".bin", stat.S_IFREG | 0o444)
            w.data = memoryview(tensor).cast("B")
            parent.children[w.name] = w

            g = VNode(name.split("/")[-1] + ".grad", stat.S_IFREG | 0o666)
            g.data = bytearray()
            parent.children[g.name] = g
    else:
        # Handle PyTorch model
        for name, p in source.items():
            parent = modeld
            for part in name.split(".")[:-1]:
                parent = parent.children.setdefault(part, VNode(part, stat.S_IFDIR | 0o755))

            w = VNode(name.split(".")[-1] + ".bin", stat.S_IFREG | 0o444)
            w.data = memoryview(p.detach().cpu().numpy()).cast("B")
            parent.children[w.name] = w

            g = VNode(name.split(".")[-1] + ".grad", stat.S_IFREG | 0o666)
            g.data = bytearray()
            parent.children[g.name] = g

    root.children["activations"] = VNode("activations", stat.S_IFDIR | 0o755)

    ird = VNode("ir", stat.S_IFDIR | 0o444); root.children["ir"] = ird
    ird.children["graph.txt"] = VNode("graph.txt", stat.S_IFREG | 0o444)

    logd = VNode("logs", stat.S_IFDIR | 0o444); root.children["logs"] = logd
    logd.children["fuse.log"] = VNode("fuse.log", stat.S_IFREG | 0o444)

    return root

# ───────── FUSE ops ─────────────────────────────────────────────────────────
class MLFS(Operations):
    def __init__(self, ckpt_path, allow_bin_write=False, load_state_dict=False):
        self.rwlock = RLock()
        self.allow_bin_write = allow_bin_write

        if ckpt_path.endswith('.onnx'):
            model = onnx.load(ckpt_path)
            self.root = build_tree(model, meta_str=f"ONNX model: {model.ir_version}")
            logging.info("Loaded ONNX model %s", ckpt_path)
        elif load_state_dict:
            weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.root = build_tree(weights, meta_str=f"state_dict:{len(weights)} tensors")
            logging.info("Loaded state_dict %s (%d tensors)", ckpt_path, len(weights))
        else:
            model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.root = build_tree(dict(model.named_parameters()), meta_str=str(model))
            logging.info("Loaded model %s", ckpt_path)

    # path resolve
    def _resolve(self, path):  # -> VNode
        node = self.root
        for part in filter(None, path.strip("/").split("/")):
            if not stat.S_ISDIR(node.mode): raise FuseOSError(errno.ENOTDIR)
            node = node.children.get(part)
            if node is None: raise FuseOSError(errno.ENOENT)
        return node

    # basic FUSE callbacks
    def getattr(self, p, fh=None):
        with self.rwlock: return self._resolve(p).stat()

    def readdir(self, p, fh):
        with self.rwlock:
            n = self._resolve(p)
            if not stat.S_ISDIR(n.mode): raise FuseOSError(errno.ENOTDIR)
            return [".", ".."] + list(n.children)

    def read(self, p, size, off, fh):
        with self.rwlock:
            buf = self._resolve(p).data
            return bytes(buf[off: off+size])

    # predicates
    _is_bin  = staticmethod(lambda p: p.startswith("/model") and p.endswith(".bin"))
    _is_grad = staticmethod(lambda p: p.startswith("/model") and p.endswith(".grad"))

    def write(self, p, buf, off, fh):
        with self.rwlock:
            n = self._resolve(p)
            if self._is_grad(p) or (self.allow_bin_write and self._is_bin(p)):
                if not isinstance(n.data, bytearray):
                    n.data = bytearray(n.data)
                n.data[off:off+len(buf)] = buf
                return len(buf)
            if p == "/logs/fuse.log":
                n.data += buf; return len(buf)
            raise FuseOSError(errno.EPERM)

    def open(self, p, flags):
        if flags & os.O_TRUNC: self.truncate(p, 0)
        return 0

    def truncate(self, p, length, fh=None):
        with self.rwlock:
            n = self._resolve(p)
            if length == 0 and (self._is_grad(p) or (self.allow_bin_write and self._is_bin(p))):
                n.data = bytearray()
                return 0
            raise FuseOSError(errno.EPERM)

    flush = release = lambda *a, **k: 0

# ───────── CLI ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MLFS – mount a model filesystem")
    ap.add_argument("--model", required=True, help=".pt/.pth or state_dict file")
    ap.add_argument("--mount", required=True)
    ap.add_argument("-f", "--foreground", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--unsafe-write", action="store_true")
    ap.add_argument("--state-dict", action="store_true",
                    help="treat checkpoint as raw state_dict (faster)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[mlfs] %(message)s")
    if args.foreground and args.wait:
        logging.warning("--wait ignored in foreground mode.")
        args.wait = False

    FUSE(
        MLFS(args.model,
             allow_bin_write=args.unsafe_write,
             load_state_dict=args.state_dict),
        args.mount,
        foreground=args.foreground,
        ro=False,
        allow_other=True,
    )

    if args.wait:
        for _ in range(120):
            if os.path.isdir(os.path.join(args.mount, "model")): break
            time.sleep(0.1)

if __name__ == "__main__":
    main()
