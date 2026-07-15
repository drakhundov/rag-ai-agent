"""Deals with loading documents from the file system."""

from typing import List
import enum
import os
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger: logging.Logger = logging.getLogger(__name__)


class DocFmt(enum.Enum):
    TXT = ".txt"
    PDF = ".pdf"
    MD = ".md"
    HTML = ".html"


ALL_AVAILABLE_FMTS = [
    DocFmt.TXT.value,
    DocFmt.PDF.value,
]


class DocumentLoader:
    def __init__(self, allowed_fmts: List = ALL_AVAILABLE_FMTS):
        self.allowed_fmts = [self._normalize_fmt(fmt) for fmt in allowed_fmts]

    @staticmethod
    def _normalize_fmt(fmt) -> str:
        if isinstance(fmt, DocFmt):
            return fmt.value
        return str(fmt)

    def load_from_fs(self, lst: List[Path] | List[str]) -> List[Document]:
        docs = []
        for fpath in lst:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Can't load document: couldn't locate path {fpath}")
            docs.extend(self._load_doc_from_fs(Path(fpath)))
        return docs

    def _load_doc_from_fs(self, fpath: Path) -> List[Document]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Can't load document: couldn't locate path {fpath}")

        suffix = fpath.suffix.lower()
        if suffix not in self.allowed_fmts:
            raise ValueError(f"DocumentLoader: format {suffix} is not allowed in this object")

        logger.debug(f"Loading document from {fpath}")
        if suffix == ".pdf":
            loader = PyPDFLoader(str(fpath))
            docs =  loader.load()
        elif suffix in {".txt", ".md", ".html"}:
            docs = [Document(page_content=fpath.read_text(encoding="utf-8"))]
        else:
            raise ValueError(
                f"Unsupported file type: {fpath.suffix}. Only {','.join(self.allowed_fmts)} formats are supported and allowed"
            )
        for doc in docs:
            doc.metadata["source"] = str(fpath)
        return docs
