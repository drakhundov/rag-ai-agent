import os
from typing import List
import logging

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from chain import OpenAIChatModel, ChromaDocumentRetriever, SemanticTextSplitter
from core.config import load_conf
from services.RAGEngine import RAGEngine

logger: logging.Logger = logging.getLogger(__name__)

# Ensure imports could omit "src".
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_langsmith():
    with load_conf() as conf:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = conf.langchain_api_key.get_secret_value()
        os.environ["LANGCHAIN_PROJECT"] = "RAG"
        os.environ["LANGSMITH_ENDPOINT"] = conf.paths.langsmith_api_url
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    
def build_rag_engine(filepaths: List[str]) -> RAGEngine:
    docs = []
    for fpath in filepaths:
        logger.debug(f"Loading document from {fpath}")
        loader = PyPDFLoader(fpath)
        docs.extend(loader.load())
    doc_retriever = ChromaDocumentRetriever(
        docs=docs,
        text_splitter=SemanticTextSplitter()
        # Use default config.
        # Use default chroma index directory.
    )
    chat_model = OpenAIChatModel()  # Use default config.
    with load_conf() as conf:
        sys_prompt_template = PromptTemplate(
            input_variables=conf.prompt_templs.system.input_variables,
            template=conf.prompt_templs.system.template
        )
    return RAGEngine(
        doc_retriever=doc_retriever, chat_model=chat_model, sys_prompt_template=sys_prompt_template
    )
