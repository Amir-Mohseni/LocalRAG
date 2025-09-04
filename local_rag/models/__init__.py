"""Models package for LocalRAG."""

from .local_llm import LocalLLM
from .local_embedding import LocalEmbedding

__all__ = ["LocalLLM", "LocalEmbedding"]
