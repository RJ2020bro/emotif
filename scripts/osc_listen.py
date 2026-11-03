#!/usr/bin/env python3
"""Print the OSC messages the Emotif server sends (check the bridge before opening Unreal Engine).

  python -m emotif.server --osc 127.0.0.1:8001      # terminal 1
  python scripts/osc_listen.py 8001                  # terminal 2
"""
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotif import osc  # noqa: E402

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("127.0.0.1", port))
print(f"listening on 127.0.0.1:{port}")
while True:
    addr, args = osc.parse(s.recv(1024))
    print(addr, " ".join(f"{a:+.3f}" if isinstance(a, float) else str(a) for a in args), flush=True)
