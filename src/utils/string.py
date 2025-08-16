import re
from typing import Dict

_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")


def replace_placeholders(string: str, mapping: Dict[str, str]) -> str:
    def repl(_match: str):
        var_name = _match.group(1)
        return str(mapping.get(var_name, f"${{{var_name}}}"))

    return _PLACEHOLDER_RE.sub(repl, string)
