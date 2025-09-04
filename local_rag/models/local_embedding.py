"""Local embedding implementation for OpenAI-compatible API endpoints."""

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI, AsyncOpenAI
from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalEmbedding(BaseEmbedding):
    """Custom embedding implementation for local OpenAI-compatible API endpoints."""
    
    def __init__(
        self, 
        client: OpenAI, 
        async_client: AsyncOpenAI, 
        model_name: str, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = client
        self._async_client = async_client
        self._model_name = model_name
    
    def _filter_embedding_kwargs(self, kwargs: dict) -> dict:
        """Filter out unsupported parameters for embedding API calls."""
        # List of supported OpenAI embedding parameters
        supported_params = {
            'user', 'encoding_format', 'dimensions'
        }
        
        # Filter kwargs to only include supported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        # Log any filtered parameters for debugging
        filtered_out = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out:
            logger.debug(f"Filtered out unsupported embedding parameters: {filtered_out}")
        
        return filtered_kwargs
    
    def _get_query_embedding(self, query: str) -> List[float]:
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=query,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        try:
            response = await self._async_client.embeddings.create(
                model=self._model_name,
                input=query,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting async query embedding: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            raise
