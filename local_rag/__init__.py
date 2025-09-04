"""LocalRAG - A local RAG system using LlamaIndex with local models."""

from .models.local_llm import LocalLLM
from .models.local_embedding import LocalEmbedding
from .utils.config import load_config
from .rag_system import create_local_rag_system, interactive_query_loop
from .web_interface import create_web_interface, LocalRAGChatInterface

__version__ = "1.0.0"
__all__ = [
    "LocalLLM",
    "LocalEmbedding", 
    "load_config",
    "create_local_rag_system",
    "interactive_query_loop",
    "create_web_interface",
    "LocalRAGChatInterface"
]
