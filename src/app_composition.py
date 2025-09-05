import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from config import load_conf

from adapters import ChatOpenAILLMService, ChromaRetriever, SemanticTextSplitter
from services.retriever_service import RetrieverService

# Ensure imports could omit "src".
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def build_app() -> RetrieverService:
    conf = load_conf()
    # Set up LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = conf.langchain_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = "RAG"
    os.environ["LANGSMITH_ENDPOINT"] = conf.paths.langsmith_api_url
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    retriever = ChromaRetriever(
        pdf_paths=["context.pdf"],
        emb_model=HuggingFaceEmbeddings(model_name=conf.models.emb_model_name),
        text_splitter=SemanticTextSplitter(),  # Use default config.
        # Use default chroma index directory.
    )
    llm_model = ChatOpenAILLMService()  # Use default config.
    sys_prompt_template = PromptTemplate(
        input_variables=conf.prompt_templs.system.input_variables,
        template=conf.prompt_templs.system.template,
    )
    return RetrieverService(
        retriever=retriever, llm_model=llm_model, sys_prompt_template=sys_prompt_template
    )
