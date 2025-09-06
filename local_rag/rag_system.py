"""Main RAG system implementation."""

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from openai import OpenAI, AsyncOpenAI
import logging
from typing import Optional

from .models.local_llm import LocalLLM
from .models.local_embedding import LocalEmbedding
from .utils.config import load_config, get_nested_config, generate_system_prompt
from .utils.model_manager import ModelManager

logger = logging.getLogger(__name__)


def create_local_rag_system(config_path: str = "config.yaml", force_check: bool = False) -> Optional[VectorStoreIndex]:
    """
    Create a complete RAG system with local models using configuration file.
    
    Args:
        config_path: Path to the configuration file
        force_check: If True, always check if configured models are loaded
        
    Returns:
        VectorStoreIndex: Configured vector store index
        
    Raises:
        RuntimeError: If configured models are not loaded in LM Studio
    """
    # Load configuration first
    config = load_config(config_path)
    log_level = get_nested_config(config, "logging.level", "INFO")
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    # Check if setup is completed and models are configured
    model_manager = ModelManager(config_path)
    setup_completed = model_manager.is_setup_completed()
    
    if not setup_completed or force_check:
        # Check if configured models are actually loaded
        configured_models = model_manager.get_configured_models()
        loaded_models = ModelManager.get_loaded_models()
        
        # Validate that configured models are loaded
        llm_loaded = configured_models.get('llm') and configured_models['llm'] in loaded_models['llm']
        embedding_loaded = configured_models.get('embedding') and configured_models['embedding'] in loaded_models['embedding']
        
        if not llm_loaded or not embedding_loaded:
            available_models = ModelManager.get_available_models()
            
            error_msg = "❌ Configured models are not loaded in **LM Studio**!\n\n"
            
            if configured_models.get('llm'):
                error_msg += f"Configured LLM: '{configured_models['llm']}' - {'✅ Loaded' if llm_loaded else '❌ Not loaded'}\n"
            if configured_models.get('embedding'):
                error_msg += f"Configured embedding: '{configured_models['embedding']}' - {'✅ Loaded' if embedding_loaded else '❌ Not loaded'}\n\n"
            
            error_msg += "Please load the configured models in **LM Studio**:\n"
            error_msg += "1. Open **LM Studio**\n"
            error_msg += "2. Go to the 'Local Server' tab\n"
            error_msg += "3. Load the configured models\n\n"
            
            if available_models["llm"]:
                error_msg += f"Available LLM models: {', '.join(available_models['llm'])}\n"
            if available_models["embedding"]:
                error_msg += f"Available embedding models: {', '.join(available_models['embedding'])}\n"
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Extract configuration values
    base_url = get_nested_config(config, "api.base_url", "http://127.0.0.1:1234/v1")
    api_key = get_nested_config(config, "api.api_key", "not-needed")
    
    llm_model = get_nested_config(config, "models.llm.model_name", "")
    context_window = get_nested_config(config, "models.llm.context_window", 4096)
    num_output = get_nested_config(config, "models.llm.num_output", 1024)
    
    embedding_model = get_nested_config(config, "models.embedding.model_name", "")
    
    # Only initialize models if they are configured and loaded
    if llm_model and embedding_model:
        # Initialize clients
        local_client = OpenAI(base_url=base_url, api_key=api_key)
        async_local_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        # Generate system prompt with current date/time
        system_prompt = generate_system_prompt()
        
        # Configure LLM
        Settings.llm = LocalLLM(
            client=local_client,
            async_client=async_local_client,
            model_name=llm_model,
            context_window=context_window,
            num_output=num_output,
            system_prompt=system_prompt
        )
        logger.info("✅ LLM configured with local client")
        
        # Configure embedding model
        Settings.embed_model = LocalEmbedding(
            client=local_client,
            async_client=async_local_client,
            model_name=embedding_model
        )
        logger.info("✅ Embedding model configured with local client")
        
        # Create empty index (only when models are properly configured)
        try:
            logger.info("Creating empty index... (Using local embedding model)")
            # Create a placeholder document to initialize the index
            placeholder_doc = Document(text="Welcome! Please upload documents to get started.")
            index = VectorStoreIndex.from_documents([placeholder_doc])
            logger.info("✅ Empty index created successfully")
            return index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    else:
        # Return None when models aren't configured - web interface will handle this
        logger.info("Models not configured - returning None for setup")
        return None


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
