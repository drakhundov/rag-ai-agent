"""This module deals with .env (API TOKENS) and configuration (models, paths)."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import dotenv
from pydantic import SecretStr

from utilities.string import replace_placeholders


@dataclass(frozen=True)
class _PromptTempl:
    input_variables: List[str]
    template: str


@dataclass(frozen=True)
class _PromptTempls:
    system: _PromptTempl
    multi_query_rag_prompt: _PromptTempl
    hyde_rag_prompt: _PromptTempl
    decomposition_rag_prompt: _PromptTempl
    step_up_rag_prompt: _PromptTempl


@dataclass(frozen=True)
class _Models:
    chat_model_name: str
    emb_model_name: str


@dataclass(frozen=True)
class _Paths:
    proj_dir: Path
    cache_dir: Path
    chroma_index_dir: Path
    router_url: str
    langsmith_api_url: str


@dataclass(frozen=True)
class _Config:
    openai_api_key: SecretStr
    hf_token: SecretStr
    langchain_api_key: SecretStr
    models: _Models
    paths: _Paths
    prompt_templs: _PromptTempls
    concat_bufsz: int
    breakpoint_percentile_threshold: int

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# `lru_cache` caches function output values, so that
# next time the function is called with the same
# arguments, the value is returned from cache.
# `maxsize=1` tells to only store one args-return_val pair.
@lru_cache(maxsize=1)
def load_conf() -> _Config:
    # This file's path: PROJ_DIR/src/config.py
    # Thus, we need the second parent.
    proj_dir = Path(__file__).resolve().parents[1]
    # Load variables from the '.env' file.
    if not os.path.exists(proj_dir / ".env"):
        raise FileNotFoundError(f"File `{proj_dir / '.env'}` does not exist.")
    dotenv.load_dotenv(proj_dir / ".env", override=False)

    if (hf_token := os.getenv("HF_TOKEN")) is None:
        raise ValueError("You must define an HF_TOKEN in the `.env` file.")
    if (openai_api_key := os.getenv("OPENAI_API_KEY")) is None:
        raise ValueError("You must define an OPENAI_API_KEY in the `.env` file.")
    if (langchain_api_key := os.getenv("LANGCHAIN_API_KEY")) is None:
        raise ValueError("You must define an LANGCHAIN_API_KEY in the `.env` file.")

    if not os.path.exists(proj_dir / "settings.json"):
        raise FileNotFoundError(f"File `{proj_dir / 'settings.json'}` does not exist.")
    with open(proj_dir / "settings.json") as fp:
        settings = {**json.load(fp), **{"PROJ_DIR": proj_dir}}
    resolved = dict(settings)
    for k, v in resolved.items():
        if isinstance(v, str):
            resolved[k] = replace_placeholders(v, resolved)
    paths = _Paths(
        proj_dir=proj_dir,
        cache_dir=Path(resolved["CACHE_DIR"]).resolve(),
        chroma_index_dir=Path(resolved["CHROMA_INDEX_DIR"]).resolve(),
        router_url=resolved["ROUTER_URL"],
        langsmith_api_url=resolved["LANGSMITH_API_URL"]
    )
    models = _Models(
        chat_model_name=resolved["MODELS"]["DEFAULT_CHAT_MODEL"],
        emb_model_name=resolved["MODELS"]["DEFAULT_EMB_MODEL"]
    )
    prompt_templs = _PromptTempls(
        system=_PromptTempl(
            input_variables=resolved["PROMPT_TEMPLATES"]["SYSTEM"]["INPUT_VARIABLES"],
            template=resolved["PROMPT_TEMPLATES"]["SYSTEM"]["TEMPLATE"],
        ),
        multi_query_rag_prompt=_PromptTempl(
            input_variables=resolved["PROMPT_TEMPLATES"]["MULTI_QUERY_RAG_PROMPT"]["INPUT_VARIABLES"],
            template=resolved["PROMPT_TEMPLATES"]["MULTI_QUERY_RAG_PROMPT"]["TEMPLATE"],
        ),
        hyde_rag_prompt=_PromptTempl(
            input_variables=resolved["PROMPT_TEMPLATES"]["HYDE_RAG_PROMPT"]["INPUT_VARIABLES"],
            template=resolved["PROMPT_TEMPLATES"]["HYDE_RAG_PROMPT"]["TEMPLATE"],
        ),
        decomposition_rag_prompt=_PromptTempl(
            input_variables=resolved["PROMPT_TEMPLATES"]["DECOMPOSITION_RAG_PROMPT"]["INPUT_VARIABLES"],
            template=resolved["PROMPT_TEMPLATES"]["DECOMPOSITION_RAG_PROMPT"]["TEMPLATE"],
        ),
        step_up_rag_prompt=_PromptTempl(
            input_variables=resolved["PROMPT_TEMPLATES"]["STEP_UP_RAG_PROMPT"]["INPUT_VARIABLES"],
            template=resolved["PROMPT_TEMPLATES"]["STEP_UP_RAG_PROMPT"]["TEMPLATE"],
        )
    )
    return _Config(
        openai_api_key=SecretStr(openai_api_key),
        hf_token=SecretStr(hf_token),
        langchain_api_key=SecretStr(langchain_api_key),
        models=models,
        paths=paths,
        prompt_templs=prompt_templs,
        concat_bufsz=resolved["SENTENCE_CONCAT_BUFSZ"],
        breakpoint_percentile_threshold=resolved["BREAKPOINT_PERCENTILE_THRESHOLD"]
    )


__all__ = ["load_conf"]
