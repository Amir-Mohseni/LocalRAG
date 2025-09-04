"""Main RAG system implementation."""

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from openai import OpenAI, AsyncOpenAI
import logging

from .models.local_llm import LocalLLM
from .models.local_embedding import LocalEmbedding
from .utils.config import load_config, get_nested_config

logger = logging.getLogger(__name__)


def create_local_rag_system(config_path: str = "config.yaml") -> VectorStoreIndex:
    """
    Create a complete RAG system with local models using configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        VectorStoreIndex: Configured vector store index
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration values
    base_url = get_nested_config(config, "api.base_url", "http://127.0.0.1:1234/v1")
    api_key = get_nested_config(config, "api.api_key", "not-needed")
    
    llm_model = get_nested_config(config, "models.llm.model_name", "qwen/qwen3-4b-thinking-2507")
    context_window = get_nested_config(config, "models.llm.context_window", 4096)
    num_output = get_nested_config(config, "models.llm.num_output", 1024)
    
    embedding_model = get_nested_config(config, "models.embedding.model_name", "text-embedding-embeddinggemma-300m-qat")
    data_directory = get_nested_config(config, "data.directory", "./data")
    
    log_level = get_nested_config(config, "logging.level", "INFO")
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    # Initialize clients
    local_client = OpenAI(base_url=base_url, api_key=api_key)
    async_local_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # Configure LLM
    Settings.llm = LocalLLM(
        client=local_client,
        async_client=async_local_client,
        model_name=llm_model,
        context_window=context_window,
        num_output=num_output
    )
    logger.info("✅ LLM configured with local client")
    
    # Configure embedding model
    Settings.embed_model = LocalEmbedding(
        client=local_client,
        async_client=async_local_client,
        model_name=embedding_model
    )
    logger.info("✅ Embedding model configured with local client")
    
    # Load documents
    try:
        documents = SimpleDirectoryReader(data_directory).load_data()
        logger.info(f"✅ Loaded {len(documents)} document(s)")
    except Exception as e:
        logger.error(f"Error loading documents from {data_directory}: {e}")
        raise
    
    # Create index
    try:
        logger.info("Building index... (Using local embedding model)")
        index = VectorStoreIndex.from_documents(documents)
        logger.info("✅ Index built successfully")
        return index
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise


def interactive_query_loop(index: VectorStoreIndex):
    """Run an interactive query loop with the given index."""
    query_engine = index.as_query_engine()
    logger.info("✅ Query engine created. You can now ask questions!")
    
    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            response = query_engine.query(query)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")
