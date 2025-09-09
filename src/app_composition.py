import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

from config import load_conf

from adapters import OpenAIChatModel, ChromaDocumentRetriever, SemanticTextSplitter
from services.rag_engine import RAGEngine


# Ensure imports could omit "src".
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def build_app() -> RAGEngine:
    conf = load_conf()
    # Set up LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = conf.langchain_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = "RAG"
    os.environ["LANGSMITH_ENDPOINT"] = conf.paths.langsmith_api_url
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    docs = PyPDFLoader("context.pdf").load()
    doc_retriever = ChromaDocumentRetriever(
        docs=docs,
        text_splitter=SemanticTextSplitter()
        # Use default config.
        # Use default chroma index directory.
    )
    chat_model = OpenAIChatModel()  # Use default config.
    sys_prompt_template = PromptTemplate(
        input_variables=conf.prompt_templs.system.input_variables,
        template=conf.prompt_templs.system.template
    )
    return RAGEngine(
        doc_retriever=doc_retriever, chat_model=chat_model, sys_prompt_template=sys_prompt_template
    )
