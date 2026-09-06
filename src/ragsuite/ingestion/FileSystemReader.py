"""Deals with loading documents from the file system."""

import logging
import os
from pathlib import Path
from collections import deque
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ragsuite.ingestion.document_formats import DocumentFormat, FORMAT_CHECKERS
from ragsuite.utilities import err

logger: logging.Logger = logging.getLogger(__name__)

# Kept as an alias so callers/tests can `from ragsuite.ingestion import DocFmt`
# without caring that the enum itself lives in document_formats.
DocFmt = DocumentFormat

ALL_AVAILABLE_FMTS = [DocFmt.TXT.value, DocFmt.PDF.value]


class FileSystemReader:
    def __init__(self, allowed_fmts: List = ALL_AVAILABLE_FMTS, check_format: bool = False):
        """
        Args:
            allowed_fmts:
                which formats (DocFmt members or their string values, e.g. ".txt") this
                loader will accept. Defaults to txt and pdf.
            check_format:
                if True, every loaded document is run through its format's checker
                (see `ragsuite.ingestion.document_formats`) before being returned, and
                DocumentFormatError is raised on anything corrupted or malformed.
                If False (default), documents are ingested as-is without verification.
        """
        self.allowed_fmts = [self._normalize_fmt(fmt) for fmt in allowed_fmts]
        self.check_format = check_format

    @staticmethod
    def _normalize_fmt(fmt) -> str:
        if isinstance(fmt, DocFmt):
            return fmt.value
        return str(fmt)

    def load(self, flst: List[Path] | List[str] = None, dlst: List[Path] | List[str] = None, recursive: bool = False) -> List[Document]:
        """
            Args:
                flst:
                    list of file paths to load into documents.
                dlst:
                    list of directories to scrap for for files confomring to allowed_fmts.
                recursive:
                    if true, dlst will be searched recursively.
            Returns:
                list of LangChain Document objects.
            """
        if flst is None and dlist is None:
            err.log_and_raise(
                logger,
                ValueError("Must provide a list of file paths or directories to parse"),
            )
        # flst is used accross loading, where subfiles from dlst are put into flst.
        flst = [] if flst is None else flst
        # First, add files from dlst to the list, next parse all files.
        q = deque()
        q.extend([os.path.join(dpath, child) for dpath in dlst for child in os.listdir(dpath)])
        while q:
            p = q.pop()
            if os.path.isdir(p) and recursive:
                q.extend([os.path.join(p, child) for child in os.listdir(p)])
            elif os.path.isfile(p):
                flst.append(p)
        docs = []
        for fpath in flst:
            if not os.path.exists(fpath):
                err.log_and_raise(
                    logger,
                    FileNotFoundError(
                        f"Can't load document: couldn't locate path {fpath}"
                    ),
                )
            docs.extend(self._load_doc_from_fs(Path(fpath)))
        return docs

    def _load_doc_from_fs(self, fpath: Path) -> List[Document]:
        if not os.path.exists(fpath):
            err.log_and_raise(
                logger,
                FileNotFoundError(f"Can't load document: couldn't locate path {fpath}")
            )

        suffix = fpath.suffix.lower()
        if suffix not in self.allowed_fmts:
            err.log_and_raise(
                logger,
                ValueError(f"FSReader: format {suffix} is not allowed in this object")
            )

        logger.debug(f"Loading document from {fpath}")
        if suffix == DocFmt.PDF.value:
            loader = PyPDFLoader(str(fpath))
            docs = loader.load()
        elif suffix in {DocFmt.TXT.value, DocFmt.MD.value, DocFmt.HTML.value}:
            # Each file becomes exactly one Document.
            docs = [Document(page_content=fpath.read_text(encoding="utf-8"))]
        else:
            err.log_and_raise(
                logger,
                ValueError(
                    f"Unsupported file type: {fpath.suffix}. Only {','.join(self.allowed_fmts)} formats are supported and allowed"
                ),
            )

        if self.check_format:
            checker = FORMAT_CHECKERS.get(DocFmt(suffix))
            if checker is not None:
                checker.check(fpath, docs)
            else:
                err.log_and_raise(
                    logger,
                    ModuleNotFoundError(f"Could not locate checker for {suffix} format")
                )

        for doc in docs:
            doc.metadata["source"] = str(fpath)
        return docs
