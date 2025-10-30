import pickle

import pytest

import services.CacheManager as cm_module
from core.types import CacheAttr
from services.CacheManager import CacheManager


@pytest.fixture
def init_cache_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(cm_module, "CACHE_DIR", str(tmp_path))
    return CacheManager("documents")


def test_set_creates_pkl_file(init_cache_manager, tmp_path):
    cmng = init_cache_manager
    cache_key = "hash_text"
    value = ["doc1", "doc2", "doc3"]
    cmng.set(cache_key, {CacheAttr.SPLITTER: value}, write_as_binary=True)
    expected = tmp_path / "documents" / cache_key / (CacheAttr.SPLITTER.value + ".pkl")
    assert expected.exists()
    assert expected.read_bytes() == pickle.dumps(value)


def test_set_creates_text_file(init_cache_manager, tmp_path):
    cmng = init_cache_manager
    cache_key = "hash_text"
    value = ["doc1", "doc2", "doc3"]
    cmng.set(cache_key, {CacheAttr.SPLITTER: value}, write_as_binary=False)
    expected = tmp_path / "documents" / cache_key / (CacheAttr.SPLITTER.value + ".txt")
    assert expected.exists()
    assert expected.read_text() == str(value)


#
def test_values_match(init_cache_manager, tmp_path):
    cmng = init_cache_manager
    cache_key = "hash_bin"
    value = {"a": 1, "b": [1, 2, 3]}
    cmng.set(cache_key, {CacheAttr.SPLITTER: value}, write_as_binary=True)

    expected = tmp_path / "documents" / cache_key / (CacheAttr.SPLITTER.value + ".pkl")
    assert expected.exists()

    loaded = cmng.get(cache_key, CacheAttr.SPLITTER, read_as_binary=True)
    assert loaded == value


def test_set_raises_on_non_enum_key(init_cache_manager):
    cmng = init_cache_manager
    with pytest.raises(ValueError):
        cmng.set("some_key", {"not_an_enum": "value"})


def test_get_raises_file_not_found_for_missing_cache(init_cache_manager):
    cmng = init_cache_manager
    with pytest.raises(FileNotFoundError):
        cmng.get("nonexistent_cache_key", CacheAttr.SPLITTER, read_as_binary=True)
