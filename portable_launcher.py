from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8501


def port_is_available(port: int) -> bool:
    """Return True when a loopback TCP port can be reserved."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_local_port(preferred: int = DEFAULT_PORT) -> int:
    """Prefer the usual Streamlit port, otherwise request a free local port."""
    if port_is_available(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def offline_environment(port: int) -> dict[str, str]:
    """Create the process environment for a local-only Streamlit server."""
    environment = os.environ.copy()
    environment.update(
        {
            "CUS_AI_OFFLINE": "1",
            "PYTHONUTF8": "1",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
            "STREAMLIT_SERVER_PORT": str(port),
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        }
    )
    return environment


def streamlit_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_ROOT / "app.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]


def app_is_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.75) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def check_installation() -> int:
    required_modules = (
        "streamlit",
        "numpy",
        "PIL",
        "pydicom",
        "cv2",
        "cus_ai.clinical",
        "cus_ai.media",
        "cus_ai.model",
        "cus_ai.reporting",
    )
    failures: list[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")
    if failures:
        print("Portable installation check failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("Portable installation check passed.")
    return 0


def run_app(open_browser: bool = True) -> int:
    port = choose_local_port()
    local_url = f"http://127.0.0.1:{port}"
    print("CUS AI Reader local offline edition")
    print(f"Starting at {local_url}")
    print("Close this window or press Ctrl+C to stop the app.")

    process = subprocess.Popen(
        streamlit_command(port),
        cwd=APP_ROOT,
        env=offline_environment(port),
    )
    deadline = time.monotonic() + 90
    while process.poll() is None and time.monotonic() < deadline:
        if app_is_ready(port):
            if open_browser:
                webbrowser.open(local_url)
            break
        time.sleep(0.25)
    else:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        print("The local app did not become ready within 90 seconds.")
        return 1

    try:
        return int(process.wait())
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local-only CUS AI Reader")
    parser.add_argument("--check", action="store_true", help="verify portable dependencies and exit")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default web browser")
    args = parser.parse_args()
    if args.check:
        return check_installation()
    return run_app(open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
