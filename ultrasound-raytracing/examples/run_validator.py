#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Get the directory of the probe_validator_server.py script
validator_dir = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    / "ultrasound_sweep"
)


def main():
    """
    Run the ultrasound probe geometry validator server and open the web interface.
    """
    print("Starting Ultrasound Probe Geometry Validator...")

    # Ensure template and static folders exist
    os.makedirs(validator_dir / "templates", exist_ok=True)
    os.makedirs(validator_dir / "static", exist_ok=True)

    # Start the server process
    server_script = validator_dir / "probe_validator_server.py"

    # Make the server script executable
    os.chmod(server_script, 0o755)

    # Start the server in a subprocess
    server_process = subprocess.Popen([sys.executable, str(server_script)])

    # Open the web browser
    print("Opening web browser...")
    webbrowser.open("http://localhost:8080")

    try:
        # Wait for the server to be terminated
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()

    print("Server stopped.")


if __name__ == "__main__":
    main()
