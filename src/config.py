"""This module deals with .env (API TOKENS) and configuration (models, paths)."""

import os, dotenv, json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from utils.string import replace_placeholders


@dataclass(frozen=True)
class PromptTempl:
    input_variables: List[str]
    template: str


@dataclass(frozen=True)
class Models:
    chat_model_name: str
    emb_model_name: str


@dataclass(frozen=True)
class Paths:
    proj_dir: Path
    cache_dir: Path
    chroma_index_dir: Path
    router_url: str


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    hf_token: str
    models: Models
    paths: Paths
    prompt_templ: Dict[str:List, str:str]
    concat_bufsz: int
    breakpoint_percentile_threshold: int


# `lru_cache` caches function output values, so that
# next time the function is called with the same
# arguments, the value is returned from cache.
# `maxsize=1` tells to only store one args-return_val pair.
@lru_cache(maxsize=1)
def load_conf() -> Config:
    # This file's path: PROJ_DIR/src/config.py
    # Thus, we need the second parent.
    proj_dir = Path(__file__).resolve().parents[1]
    # Load variables from the '.env' file.
    if not os.path.exists(proj_dir / ".env"):
        raise FileNotFoundError(f"File `{proj_dir / '.env'}` does not exist.")
    dotenv.load_dotenv(proj_dir / ".env", override=False)

    hf_token = os.getenv("HF_TOKEN")
    if (openai_api_key := os.getenv("OPENAI_API_KEY")) is None:
        raise ValueError("You must define an OPENAI_API_KEY in the `.env` file.")
    if not os.path.exists(proj_dir / "settings.json"):
        raise FileNotFoundError(f"File `{proj_dir / 'settings.json'}` does not exist.")
    with open(proj_dir / "settings.json") as fp:
        settings = {**json.load(fp), **{"PROJ_DIR": proj_dir}}
    resolved = dict(settings)
    for k, v in resolved.items():
        if isinstance(v, str):
            resolved[k] = replace_placeholders(v, resolved)
    paths = Paths(
        proj_dir=proj_dir,
        cache_dir=Path(resolved["CACHE_DIR"]).resolve(),
        chroma_index_dir=Path(resolved["CHROMA_INDEX_DIR"]).resolve(),
        router_url=resolved["ROUTER_URL"],
    )
    models = Models(
        chat_model_name=resolved["MODELS"]["DEFAULT_CHAT_MODEL"],
        emb_model_name=resolved["MODELS"]["DEFAULT_EMB_MODEL"],
    )
    prompt_templ = PromptTempl(
        input_variables=resolved["PROMPT_TEMPL"]["INPUT_VARIABLES"],
        template=resolved["PROMPT_TEMPL"]["TEMPLATE"],
    )
    return Config(
        openai_api_key=openai_api_key,
        hf_token=hf_token,
        models=models,
        paths=paths,
        prompt_templ=prompt_templ,
        concat_bufsz=resolved["SENTENCE_CONCAT_BUFSZ"],
        breakpoint_percentile_threshold=resolved["BREAKPOINT_PERCENTILE_THRESHOLD"],
    )
