import logging
import os
from typing import List

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import load_conf
from core.ports import TextSplitter
from core.types import QueryStr

# TODO: cache which documents have been indexed etc.

logger: logging.Logger = logging.getLogger(__name__)


# Interface: ports/DocumentRetriever
class ChromaDocumentRetriever:
    def __init__(
        self,
        docs: List[Document],
        text_splitter: TextSplitter,
        chroma_index_dir: str = None,
        emb_model: Embeddings = None,
    ):
        logger.debug("Starting ChromaDocumentRetriever initialization")
        with load_conf() as conf:
            if chroma_index_dir is None:
                chroma_index_dir = str(conf.paths.chroma_index_dir)
            if emb_model is None:
                emb_model = HuggingFaceEndpointEmbeddings(
                    model=conf.models.emb_model_name,
                    huggingfacehub_api_token=conf.hf_token.get_secret_value(),
                )
        self.persist_dir = chroma_index_dir
        self.emb_model = emb_model
        self.text_splitter = text_splitter
        self.docs = docs
        self.vs = None
        self._initialize_index()
        logger.debug("ChromaDocumentRetriever initialized")

    def _initialize_index(self):
        """Initialize the vectorstore if hasn't been initialized yet."""
        if self.vs is not None:
            return
        chunks = self.text_splitter.split(self.docs)
        if not os.path.exists(self.persist_dir):
            self.vs = Chroma.from_documents(
                chunks, self.emb_model, persist_directory=self.persist_dir
            )
        else:
            self.vs = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.emb_model,
            )
        self.add_docs(chunks)

    def retrieve(self, query: QueryStr, top_k: int = 4) -> List[Document]:
        logger.debug(f"Retrieving {query}")
        return self.vs.as_retriever(search_kwargs={"k": top_k}).invoke(query)

    def add_docs(self, docs: List[Document]):
        logger.debug(f"Adding {len(docs)} documents to the retriever")
        self.vs.add_documents(docs)
        if callable(getattr(self.vs, "persist", None)):
            self.vs.persist()

    # ! Used to make sure compatibility with LangChain pipelines.
    def invoke(self, input, *args, **kwargs):
        return self.vs.as_retriever(kwargs).invoke(input, *args, **kwargs)
