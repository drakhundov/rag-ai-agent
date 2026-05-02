import logging
import os
from pathlib import Path
from typing import List
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from chain import OpenAIChatModel, ChromaDocumentRetriever, SemanticTextSplitter
from core.config import load_conf
from services.RAGEngine import RAGEngine
from utilities import cli

logger: logging.Logger = logging.getLogger()


# Ensure imports could omit "src".
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _load_documents_from_file(fpath: str) -> List[Document]:
    path = Path(fpath)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if suffix in {".txt", ".text", ".md"}:
        return [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)},
            )
        ]

    raise ValueError(
        f"Unsupported file type: {path.suffix}. Only PDF and text files are supported."
    )


def setup_langsmith():
    with load_conf() as conf:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = conf.langchain_api_key.get_secret_value()
        os.environ["LANGCHAIN_PROJECT"] = "RAG"
        os.environ["LANGSMITH_ENDPOINT"] = conf.paths.langsmith_api_url
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


@cli.with_temp_message(message="Initializing logs...")
def init_logs() -> logging.Logger:
    with load_conf() as conf:
        LOGS_DIR = conf.paths.logs_dir
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    LOGS_DIR, datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S.log")
                ),
                mode="a",
                encoding="utf-8",
            ),
        ],
        force=True,
    )
    return logging.getLogger()


@cli.with_temp_message(message="Building RAG Engine...")
def build_rag_engine(filepaths: List[str]) -> RAGEngine:
    docs = []
    for fpath in filepaths:
        logger.debug(f"Loading document from {fpath}")
        docs.extend(_load_documents_from_file(fpath))
    doc_retriever = ChromaDocumentRetriever(
        docs=docs,
        text_splitter=SemanticTextSplitter(),
        # Use default config.
        # Use default chroma index directory.
    )
    chat_model = OpenAIChatModel()  # Use default config.
    with load_conf() as conf:
        sys_prompt_template = PromptTemplate(
            input_variables=conf.prompt_templs.system.input_variables,
            template=conf.prompt_templs.system.template,
        )
    return RAGEngine(
        doc_retriever=doc_retriever,
        chat_model=chat_model,
        sys_prompt_template=sys_prompt_template,
    )
