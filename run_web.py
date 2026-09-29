"""
Google ADK Web Chat Interface Launcher
======================================
Starts the official Google ADK Web UI server to interact with the
Memory-Graph Chatbot and monitor TypeSafe Jev gating in real time.

Usage:
  python run_web.py [--port 8000] [--host 127.0.0.1]
"""

import sys
import os
import subprocess
import argparse
from dotenv import load_dotenv

load_dotenv()

def launch_adk_web(host: str = "127.0.0.1", port: int = 8000):
    print("================================================================================")
    print("        LAUNCHING GOOGLE ADK WEB CHAT INTERFACE WITH TYPESAFE JEV GATE          ")
    print("================================================================================")
    print(f"[*] Agent App : memory_app (Model: gemini-flash-lite-latest)")
    print(f"[*] Gate      : TypeSafe Jev (System One Zero-Gen Decision Engine)")
    print(f"[*] Web Server: http://{host}:{port}")
    print(f"[*] Direct URL: http://localhost:{port}")
    print("--------------------------------------------------------------------------------")
    print("Press Ctrl+C in terminal to stop the web server.")
    print("================================================================================\n")

    # Command to run official Google ADK Web Server for memory_app
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "google.adk.cli",
        "web",
        "memory_app",
        "--host",
        host,
        "--port",
        str(port)
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[!] Stopping Google ADK Web Chat Server...")
    except Exception as e:
        print(f"[-] Error launching ADK web server: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Google ADK Web Chat Interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    args = parser.parse_args()
    launch_adk_web(host=args.host, port=args.port)
