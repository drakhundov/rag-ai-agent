import re
from typing import Dict, List

_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")
_SPLIT_SENTENCES_RE = re.compile(r"(?<=[.!?])\s+")


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


def split_into_sentences(text: str) -> List[str]:
    """
    Splits a given text into sentences based on punctuation and removes extra spaces.

    The text is split at punctuation marks (e.g., '.', '!', '?') followed by whitespace.
    Empty or whitespace-only sentences are excluded from the result.

    Args:
        text (str): The input text to split into sentences.

    Returns:
        List[str]: A list of cleaned sentences.
    """
    if not text:
        return []
    parts = _SPLIT_SENTENCES_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def windowed_concat(sentences: List[str], bufsz: int) -> List[str]:
    """
    Concatenates a window of surrounding sentences for each sentence in the input list.

    For each sentence, this function includes up to `bufsz` sentences to the left and right,
    concatenating them into a single string. At the edges of the list, the window is adjusted
    to include only available sentences.

    Args:
        sentences (List[str]): A list of sentences to process.
        bufsz (int): The number of surrounding sentences to include on each side.

    Returns:
        List[str]: A list of concatenated strings, one for each sentence in the input list.

    Raises:
        ValueError: If `bufsz` is negative.
    """
    if bufsz < 0:
        raise ValueError("`bufsz` must be a positive integer.")
    n = len(sentences)
    windowed = []
    for i in range(n):
        start = max(0, i - bufsz)
        end = min(n, i + bufsz + 1)
        windowed.append(" ".join(sentences[start:end]))
    return windowed


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
