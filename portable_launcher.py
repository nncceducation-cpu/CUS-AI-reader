from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8501


def diagnostic_log_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "CUS-AI-reader")
    candidates.append(Path(tempfile.gettempdir()) / "CUS-AI-reader")
    for log_directory in candidates:
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            return log_directory / "startup.log"
        except OSError:
            continue
    return APP_ROOT / "CUS-AI-reader-startup.log"


def report_status(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    try:
        with diagnostic_log_path().open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    except OSError:
        pass


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
            "PYTHONDONTWRITEBYTECODE": "1",
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


def open_local_browser(local_url: str) -> bool:
    """Open the local page with the Windows shell before using Python's fallback."""
    if os.name == "nt" and hasattr(os, "startfile"):
        try:
            os.startfile(local_url)  # type: ignore[attr-defined]
            report_status("Opened the local page with the Windows default browser.")
            return True
        except OSError as exc:
            report_status(f"Windows could not open the default browser: {exc}")
    try:
        opened = bool(webbrowser.open(local_url, new=2))
    except Exception as exc:
        report_status(f"Python browser fallback failed: {exc}")
        return False
    if opened:
        report_status("Opened the local page with the Python browser fallback.")
    return opened


def check_installation() -> int:
    required_modules = (
        "streamlit",
        "pandas",
        "numpy",
        "PIL",
        "pydicom",
        "cv2",
        "onnxruntime",
        "cus_ai.agreement",
        "cus_ai.ai_consensus",
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


def run_app(open_browser: bool = True, exit_after_ready: bool = False) -> int:
    if check_installation() != 0:
        report_status(
            "The offline installation is incomplete. Do not skip ZIP extraction errors. "
            "Delete this copy and extract the complete package to a short folder such as C:\\CUSAI."
        )
        return 1

    port = choose_local_port()
    local_url = f"http://127.0.0.1:{port}"
    report_status("CUS AI Reader local offline edition")
    report_status(f"Starting at {local_url}")
    report_status(f"Startup log: {diagnostic_log_path()}")
    report_status("Close this window or press Ctrl+C to stop the app.")

    process = subprocess.Popen(
        streamlit_command(port),
        cwd=APP_ROOT,
        env=offline_environment(port),
    )
    deadline = time.monotonic() + 90
    ready = False
    while process.poll() is None and time.monotonic() < deadline:
        if app_is_ready(port):
            ready = True
            break
        time.sleep(0.25)

    if not ready:
        exit_code = process.poll()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if exit_code is None:
            report_status("The local app did not become ready within 90 seconds.")
        else:
            report_status(f"The local server stopped during startup with exit code {exit_code}.")
        return 1

    report_status(f"The local app is ready at {local_url}")
    if exit_after_ready:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        report_status("Full launcher startup check passed.")
        return 0

    if open_browser and not open_local_browser(local_url):
        report_status(
            "The browser did not open automatically. Double-click 'CUS AI Reader Local Page.url' "
            f"or paste {local_url} into a browser."
        )

    try:
        exit_code = int(process.wait())
        report_status(f"The local server stopped with exit code {exit_code}.")
        return exit_code
    except KeyboardInterrupt:
        report_status("Stopping the local server.")
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
    parser.add_argument(
        "--startup-check",
        action="store_true",
        help="start the local server, verify health, stop it, and exit",
    )
    args = parser.parse_args()
    if args.check:
        return check_installation()
    return run_app(open_browser=not args.no_browser, exit_after_ready=args.startup_check)


if __name__ == "__main__":
    raise SystemExit(main())

