"""
Per-format validators used by DocumentLoader to make sure a file actually
conforms to what its extension claims, instead of just trusting the suffix.

Each DocFmt gets a DocumentFormatBase subclass that knows how to look at the
Document(s) already loaded from a file and raise DocumentFormatError on
anything that looks corrupted or malformed. Checkers validate the Documents
the loader produced rather than re-opening and re-parsing the file
themselves, so a file is only ever processed once.
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar, List

from langchain_core.documents import Document

from ragsuite.utilities import err

logger: logging.Logger = logging.getLogger(__name__)


class DocumentFormat(Enum):
    TXT = ".txt"
    PDF = ".pdf"
    MD = ".md"
    HTML = ".html"

# Tags that never require a matching closing tag.
_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class DocumentFormatError(ValueError):
    """Raised when a document does not conform to the format implied by its extension."""


class DocumentFormatBase(ABC):
    """Base class for format checkers. One subclass per DocFmt."""

    fmt: ClassVar[DocumentFormat]

    def check(self, fpath: Path, docs: List[Document]) -> None:
        """Raise DocumentFormatError if `docs` (loaded from `fpath`) violate the format."""
        for doc in docs:
            self._check_one(fpath, doc)

    @abstractmethod
    def _check_one(self, fpath: Path, doc: Document) -> None: ...


class TxtFormat(DocumentFormatBase):
    """Plain text has no syntax to violate, so this is intentionally a no-op."""

    fmt = DocumentFormat.TXT

    def _check_one(self, fpath: Path, doc: Document) -> None:
        pass


class MdFormat(DocumentFormatBase):
    """Simple regex-based sanity checks for common markdown corruption."""

    fmt = DocumentFormat.MD
    _FENCE_RE = re.compile(r"^ {0,3}(```|~~~)", re.MULTILINE)

    def _check_one(self, fpath: Path, doc: Document) -> None:
        text = doc.page_content
        if len(self._FENCE_RE.findall(text)) % 2 != 0:
            msg = f"{fpath}: malformed markdown - unbalanced code fence (```/~~~) blocks"
            err.log_and_raise(logger, DocumentFormatError(msg))
        if text.count("[") != text.count("]"):
            msg = f"{fpath}: malformed markdown - unbalanced '[' / ']' (broken link or image syntax)"
            err.log_and_raise(logger, DocumentFormatError(msg))
        if text.count("(") != text.count(")"):
            msg = f"{fpath}: malformed markdown - unbalanced '(' / ')' (broken link or image syntax)"
            err.log_and_raise(logger, DocumentFormatError(msg))


class HtmlFormat(DocumentFormatBase):
    """Uses stdlib html.parser to make sure every tag is properly opened and closed."""

    fmt = DocumentFormat.HTML

    class _TagBalanceParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: List[str] = []
            self.errors: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in _VOID_ELEMENTS:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in _VOID_ELEMENTS:
                return
            if not self.stack or self.stack[-1] != tag:
                self.errors.append(f"unexpected closing tag </{tag}>")
                return
            self.stack.pop()

    def _check_one(self, fpath: Path, doc: Document) -> None:
        parser = self._TagBalanceParser()
        try:
            parser.feed(doc.page_content)
            parser.close()
        except Exception as exc:
            err.log_and_raise(logger, DocumentFormatError(f"{fpath}: malformed HTML ({exc})"))
        if parser.errors:
            msg = f"{fpath}: malformed HTML - {'; '.join(parser.errors)}"
            err.log_and_raise(logger, DocumentFormatError(msg))
        if parser.stack:
            msg = f"{fpath}: malformed HTML - unclosed tag(s): {', '.join(parser.stack)}"
            err.log_and_raise(logger, DocumentFormatError(msg))


class PdfFormat(DocumentFormatBase):
    """
    Validates the Document(s) PyPDFLoader already produced while loading the PDF.

    PyPDFLoader parses the file with pypdf under the hood and already raises on
    corrupted or encrypted PDFs during loading, before this checker ever runs.
    Re-opening the file here with a second PdfReader would just parse it twice,
    so this only checks the already-loaded result instead.
    """

    fmt = DocumentFormat.PDF

    def check(self, fpath: Path, docs: List[Document]) -> None:
        if not docs:
            err.log_and_raise(logger, DocumentFormatError(f"{fpath}: PDF contains no pages"))

    def _check_one(self, fpath: Path, doc: Document) -> None:
        # Unused: PDF is verified as a whole file in check(), not per-Document.
        pass


FORMAT_CHECKERS = {
    checker_cls.fmt: checker_cls()
    for checker_cls in (TxtFormat, MdFormat, HtmlFormat, PdfFormat)
}
