import os
from pathlib import Path

import pytest

# ! Use 'PYTHONPATH=src pytest'
from ragsuite.core.config import load_conf

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ["PROJ_DIR"] = PROJ_DIR

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
    load_conf.cache_clear()
    os.environ["PROJ_DIR"] = str(with_temp_conf)
    conf = load_conf()

    assert str(conf.paths.proj_dir) == str(with_temp_conf)
    assert isinstance(conf.paths.chroma_index_dir, Path)
    assert conf.prompt_templs.system.input_variables is not None
    assert conf.prompt_templs.system.template is not None
    assert str(conf.paths.chroma_index_dir) == os.path.join(str(with_temp_conf), "cache/chroma_index")


def test_path_resolution(with_temp_conf):
    load_conf.cache_clear()
    os.environ["PROJ_DIR"] = str(with_temp_conf)
    conf = load_conf()

    assert conf.paths.cache_dir.is_absolute()
    assert conf.paths.chroma_index_dir.is_absolute()
    assert conf.paths.hf_router_url.startswith("http")


def test_config_load_no_env(monkeypatch):
    load_conf.cache_clear()
    monkeypatch.delenv("PROJ_DIR", raising=False)
    
    conf = load_conf()
    assert str(conf.paths.proj_dir) == PROJ_DIR
