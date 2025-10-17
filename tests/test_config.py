import pytest
import os

from pathlib import Path

# ! Use 'PYTHONPATH=src pytest'
from core.config import load_conf

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

@pytest.fixture
def with_temp_conf(tmp_path, monkeypatch):
    # Prepares 'settings.json' and '.env' in a separate directory.
    monkeypatch.chdir(tmp_path)
    with open(os.path.join(PROJ_DIR, "settings.json")) as settings_f:
        (tmp_path / "settings.json").write_text(settings_f.read())
    with open(os.path.join(PROJ_DIR, ".env")) as env_f:
        (tmp_path / ".env").write_text(env_f.read())
    yield tmp_path


def test_config_load(with_temp_conf):
    os.environ["PROJ_DIR"] = str(with_temp_conf)
    conf = load_conf()

    assert str(conf.paths.proj_dir) == PROJ_DIR
    assert isinstance(conf.paths.chroma_index_dir, Path)
    assert conf.prompt_templs.system.input_variables is not None
    assert conf.prompt_templs.system.template is not None
    assert str(conf.paths.chroma_index_dir) == os.path.join(PROJ_DIR, "cache/chroma_index")


def test_path_resolution(with_temp_conf):
    os.environ["PROJ_DIR"] = str(with_temp_conf)
    conf = load_conf()

    assert conf.paths.cache_dir.is_absolute()
    assert conf.paths.chroma_index_dir.is_absolute()
    assert conf.paths.hf_router_url.startswith("http")
    assert conf.paths.langsmith_api_url.startswith("http")
