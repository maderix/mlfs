#!/usr/bin/env python3
"""
Fuse‑Chat Demo
--------------
Spins MLFS in the background and opens two terminals:

  • writer – every 2 s: echo \"ping‑<t>\" >> /logs/fuse.log
  • reader – `tail -f` the same file and print live updates

Run: sudo python3 examples/fuse_chat_demo.py
Kill both spawned xterm windows (or ctrl‑c) to exit.
"""

import os, subprocess, tempfile, shutil, time

model_stub = 'empty.pt'
import torch; torch.save({}, model_stub)   # empty state_dict

mnt = tempfile.mkdtemp(prefix='mlfs_')
proc = subprocess.Popen(
    ['sudo', 'python3', 'mlfs.py', '--model', model_stub, '--mount', mnt],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)
time.sleep(1)

log = f"{mnt}/logs/fuse.log"
open(log, 'ab').close()                   # ensure file exists

print("Opening chat windows … close them to quit.\n")

writer = subprocess.Popen(
    ['xterm', '-T', 'MLFS‑Writer',
     '-e', f"bash -c 'i=0; while true; do echo ping-$i >> {log}; i=$((i+1)); sleep 2; done'"]
)
reader = subprocess.Popen(
    ['xterm', '-T', 'MLFS‑Reader',
     '-e', f"bash -c 'tail -f {log}'"]
)

try:
    writer.wait()
finally:
    reader.terminate()
    writer.terminate()
    subprocess.run(['sudo', 'fusermount', '-u', mnt])
    proc.terminate()
    shutil.rmtree(mnt)
    os.remove(model_stub)
