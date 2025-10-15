import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import load_conf
from core.ports import TextSplitter
from core.types import QueryStr

# TODO: cache which documents have been indexed etc.


# Interface: ports/DocumentRetriever
class ChromaDocumentRetriever:
    def __init__(
        self,
        docs: List[Document],
        text_splitter: TextSplitter,
        chroma_index_dir: str = None,
        emb_model: Embeddings = None,
    ):
        with load_conf() as conf:
            if chroma_index_dir is None:
                chroma_index_dir = conf.paths.chroma_index_dir
            if emb_model is None:
                emb_model = HuggingFaceEmbeddings(model_name=conf.models.emb_model_name)
        self.persist_dir = chroma_index_dir
        self.emb_model = emb_model
        self.text_splitter = text_splitter
        self.docs = docs
        self.vs = None
        self._initialize_index()

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
        return self.vs.as_retriever(search_kwargs={"k": top_k}).invoke(query)

    def add_docs(self, docs: List[Document]):
        self.vs.add_documents(docs)
        self.vs.persist()  # Write changes to disk.

    # ! Used to make sure compatibility with LangChain pipelines.
    def invoke(self, input, *args, **kwargs):
        return self.vs.as_retriever(kwargs).invoke(input, *args, **kwargs)
