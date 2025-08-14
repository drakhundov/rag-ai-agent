"""This module deals with .env (API TOKENS) and configuration (models, paths)."""

import os, dotenv, json, re


class Config:
    def __init__(self):
        pass

    @staticmethod
    def load_conf() -> "Config":
        conf = Config()
        # Load variables from the '.env' file.
        dotenv.load_dotenv()

        conf.HF_TOKEN = os.getenv("HF_TOKEN")
        if (OPENAI_API_KEY := os.getenv("OPENAI_API_KEY")) is None:
            raise ValueError("You must define an OPENAI_API_KEY in the `.env` file.")
        else:
            conf.OPENAI_API_KEY = OPENAI_API_KEY
        conf.PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
        with open("settings.json") as fp:
            settings = json.load(fp)
        for k, v in settings.items():
            v = conf.replace_placeholders(v)
            setattr(conf, k, v)
        return conf

    def replace_placeholders(self, string: str) -> None:
        pattern = r"\$\{(\w+)\}"
        def callback(_match: str):
            var_name = _match.group(1)
            return getattr(self, var_name, f"${{{var_name}}}")
        return re.sub(pattern, callback, string)
    