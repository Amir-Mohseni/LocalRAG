#!/usr/bin/env python3
"""
Quick launcher for LocalRAG web interface.
"""

import subprocess
import sys

if __name__ == "__main__":
    # Launch web interface
    cmd = [sys.executable, "main.py", "--web"] + sys.argv[1:]
    subprocess.run(cmd)
