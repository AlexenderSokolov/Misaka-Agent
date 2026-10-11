"""pi's ``stale-socket-server.mjs``: listen on a Unix socket, report it, and wait to be killed."""

import socket
import sys
import time

if len(sys.argv) < 2 or not sys.argv[1]:
    raise SystemExit("Socket path argument is required")

server = socket.socket(socket.AF_UNIX)
server.bind(sys.argv[1])
server.listen()
print("listening", flush=True)
time.sleep(3600)
