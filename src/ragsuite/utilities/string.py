import re
import unicodedata
from typing import Dict

_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")


def normalize_input(text: str) -> str:
    """
    Standardizes and normalizes input text before processing.

    Applies the following transformations:
    1. Strip leading/trailing whitespace
    2. Normalize unicode characters (NFD normalization)
    3. Remove diacritics (accents)
    4. Convert to lowercase
    5. Collapse multiple spaces into single spaces
    6. Replace semicolons and colons with periods.
    7. Remove extra punctuation marks (but keep essential ones like ? . ! ,)

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    if not text:
        return ""

    # Strip leading/trailing whitespace
    text = text.strip()

    # Normalize unicode and remove diacritics
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Convert to lowercase
    text = text.lower()

    # Collapse multiple spaces into single space
    text = re.sub(r"\s+", " ", text)

    # Replace semicolons and colons with periods.
    text = re.sub(r";|:", ".", text)

    # Remove excessive punctuation (keep only ?, ., !, ,)
    text = re.sub(r"[^\w\s?.,!]", "", text)

    return text


def replace_placeholders(string: str, mapping: Dict[str, str]) -> str:
    """
    Replaces placeholders in the format '${VARIABLE}' in a string using a provided mapping.

    If a placeholder is not found in the mapping, it is left unchanged.

    Args:
        string (str): The input string containing placeholders.
        mapping (Dict[str, str]): A dictionary mapping placeholder names to their replacements.

    Returns:
        str: The string with placeholders replaced by their corresponding values.
    """

    def repl(_match: str):
        var_name = _match.group(1)
        return str(mapping.get(var_name, f"${{{var_name}}}"))

    return _PLACEHOLDER_RE.sub(repl, string)


def format_response(response: str) -> str:
    """
    Convert simple asterisk-based markup to ANSI terminal formatting:
    - **bold** -> bold (ANSI \033[1m)
    - *italic* -> italic (ANSI \033[3m)

    Inline code spans wrapped in backticks (`code`) are left untouched.
    """
    if response is None:
        return ""

    def _format_segment(text: str) -> str:
        # bold first (**) then single-star italics; use non-greedy matches
        text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"\033[1m{m.group(1)}\033[0m", text)
        text = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
            lambda m: f"\033[3m{m.group(1)}\033[0m",
            text,
        )
        return text

    # Preserve backtick-enclosed code spans by not formatting them
    parts = re.split(r"(`+[^`]*`+)", response)
    formatted = "".join(
        _format_segment(p) if i % 2 == 0 else p for i, p in enumerate(parts)
    )
    return formatted
