import os, sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from config import load_conf

from adapters import ChatOpenAILLMService, ChromaRetriever, SemanticTextSplitter
from services.retriever_service import RetrieverService

# Ensure imports could omit "src".
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def build_app() -> RetrieverService:
    conf = load_conf()
    retriever = ChromaRetriever(
        pdf_paths=["context.pdf"],
        emb_model=HuggingFaceEmbeddings(model_name=conf.models.emb_model_name),
        text_splitter=SemanticTextSplitter(),  # Use default config.
        # Use default chroma index directory.
    )
    llm_model = ChatOpenAILLMService()  # Use default config.
    prompt_template = PromptTemplate(
        input_variables=conf.prompt_templ.input_variables,
        template=conf.prompt_templ.template,
    )
    return RetrieverService(
        retriever=retriever, llm_model=llm_model, prompt_template=prompt_template
    )
