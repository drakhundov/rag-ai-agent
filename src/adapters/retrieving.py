import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ports import TextSplitter
from config import load_conf

# TODO: cache which documents have been indexed etc.

# Interface: ports/Retriever
class ChromaRetriever:
    def __init__(
        self,
        pdf_paths: List[str],
        emb_model: Embeddings,
        text_splitter: TextSplitter,
        chroma_index_dir: str = None,
    ):
        self.conf = load_conf()
        if chroma_index_dir is None:
            chroma_index_dir = conf.paths.chroma_index_dir
        self.persist_dir = chroma_index_dir
        self.emb_model = emb_model
        self.text_splitter = text_splitter
        self.pdf_paths = pdf_paths
        self.vs = None

    def _ensure_index(self):
        """Initialize the vectorstore if hasn't been initialized yet."""
        if self.vs is not None:
            return
        if not os.path.exists(self.persist_dir):
            # первичная индексация
            docs: list[Document] = []
            for path in self.pdf_paths:
                docs += PyPDFLoader(path).load()
            chunks = self.text_splitter.split(docs)
            self.vs = Chroma.from_documents(
                chunks, self.emb_model, persist_directory=self._persist_dir
            )
        else:
            self._vs = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embedding_fn,
            )

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        self._ensure_index()
        return self._vs.as_retriever(search_kwargs={"k": k}).invoke(query)

    # TODO: implement a method for adding new documents to the db.
    def add_doc(self, doc: Document): ...
