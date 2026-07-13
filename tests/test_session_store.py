import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch


class MockPaths:
    def __init__(self, tmp_path):
        self._tmp = Path(tmp_path)

    def __getattr__(self, name):
        if name == "sessions_dir":
            return str(self._tmp / "sessions")
        return str(self._tmp / name)


@dataclass(frozen=True)
class MockConfig:
    paths: MockPaths
    sessions_dir: str


class local_mock_conf:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def __enter__(self):
        return MockConfig(
            paths=MockPaths(self.tmp_path),
            sessions_dir=str(self.tmp_path / "sessions"),
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


from ragsuite.services.SessionStore import SessionStore


@pytest.fixture(autouse=True)
def setup_mock_config(monkeypatch, tmp_path):
    patch(
        "ragsuite.core.config.load_conf", return_value=local_mock_conf(tmp_path)
    ).start()
    # Patch the reference inside SessionStore that was imported at the top level
    monkeypatch.setattr(
        "ragsuite.services.SessionStore.load_conf", lambda: local_mock_conf(tmp_path)
    )
    return tmp_path


def test_session_store_deduces_file_paths(setup_mock_config):
    """1. Test if the sessions class properly deduces file paths."""
    svc_id = "test_svc"
    store = SessionStore(svc_id=svc_id)

    # Check if session_dir is properly deduced
    expected_dir = Path(setup_mock_config) / "sessions" / svc_id
    assert store.session_dir == expected_dir
    assert store.session_dir.exists()


def test_sessions_are_stored(setup_mock_config):
    """2. Test if the sessions are actually stored."""
    store = SessionStore(svc_id="test_svc")
    session_id = "test_session_id"
    data = {"key": "value"}

    store.dump(session_data=data, session_id=session_id)

    # Check that a JSON file was created
    expected_file = store.session_dir / f"{session_id}.json"
    assert expected_file.exists()


def test_sessions_stored_in_correct_location(setup_mock_config):
    """3. Test if the sessions are stored in the correct location."""
    store = SessionStore(svc_id="test_svc")
    data = {"key": "value"}

    # Without providing session_id, it should generate one based on timestamp
    store.dump(session_data=data)

    # Verify that a file was created in the correct directory
    files_in_dir = list(store.session_dir.iterdir())
    assert len(files_in_dir) == 1

    # The file should be in the svc_id subdirectory of sessions_dir
    created_file = files_in_dir[0]
    assert created_file.parent == store.session_dir
    assert str(store.session_dir).endswith("sessions/test_svc")
    assert created_file.suffix == ".json"


def test_sessions_stored_in_proper_format(setup_mock_config):
    """4. Test if the sessions are stored in proper format (JSON)."""
    store = SessionStore(svc_id="test_svc")
    session_id = "format_test_id"
    data = {
        "string_key": "string_value",
        "nested_dict": {"inner_key": "inner_value"},
        "list_key": ["item1", "item2"],
    }

    store.dump(session_data=data, session_id=session_id)

    expected_file = store.session_dir / f"{session_id}.json"

    # Read and parse the JSON file
    with open(expected_file, "r", encoding="utf-8") as f:
        stored_data = json.load(f)

    # Assert the stored JSON data perfectly matches the input data dictionary
    assert stored_data == data
