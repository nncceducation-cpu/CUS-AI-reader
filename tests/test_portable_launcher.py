from __future__ import annotations

from pathlib import Path

from portable_launcher import diagnostic_log_path, offline_environment, streamlit_command


def test_offline_environment_binds_to_loopback_and_disables_telemetry():
    environment = offline_environment(8765)

    assert environment["CUS_AI_OFFLINE"] == "1"
    assert environment["STREAMLIT_SERVER_ADDRESS"] == "127.0.0.1"
    assert environment["STREAMLIT_SERVER_PORT"] == "8765"
    assert environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert environment["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] == "none"


def test_streamlit_command_uses_local_only_server_settings():
    command = streamlit_command(8765)

    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8765" in command
    assert "--server.headless=true" in command
    assert "--browser.gatherUsageStats=false" in command


def test_diagnostic_log_has_a_stable_filename():
    assert isinstance(diagnostic_log_path(), Path)
    assert diagnostic_log_path().name in {"startup.log", "CUS-AI-reader-startup.log"}
