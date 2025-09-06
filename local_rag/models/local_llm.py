"""Local LLM implementation for OpenAI-compatible API endpoints."""

from llama_index.core.llms.llm import LLM
from llama_index.core.llms import CompletionResponse, LLMMetadata
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI, AsyncOpenAI
from typing import List, Any, Sequence
import logging

logger = logging.getLogger(__name__)


class LocalLLM(LLM):
    """Custom LLM implementation for local OpenAI-compatible API endpoints."""
    
    def __init__(
        self, 
        client: OpenAI, 
        async_client: AsyncOpenAI, 
        model_name: str,
        context_window: int = 4096,
        num_output: int = 1024,
        system_prompt: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = client
        self._async_client = async_client
        self._model_name = model_name
        self._context_window = context_window
        self._num_output = num_output
        self._system_prompt = system_prompt
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._num_output,
            model_name=self._model_name,
        )
    
    def _format_messages(self, messages: Sequence[ChatMessage]) -> List[dict]:
        """Convert ChatMessage objects to OpenAI format."""
        formatted_messages = []
        
        # Add system prompt if provided
        if self._system_prompt:
            formatted_messages.append({"role": "system", "content": self._system_prompt})
        
        # Add user/assistant messages
        formatted_messages.extend([{"role": msg.role.value, "content": msg.content} for msg in messages])
        
        return formatted_messages
    
    def _add_system_prompt_to_messages(self, messages: List[dict]) -> List[dict]:
        """Add system prompt to message list if not already present."""
        if not self._system_prompt:
            return messages
        
        # Check if system message already exists
        if messages and messages[0].get("role") == "system":
            return messages
        
        # Add system prompt at the beginning
        return [{"role": "system", "content": self._system_prompt}] + messages
    
    def _filter_kwargs(self, kwargs: dict) -> dict:
        """Filter out unsupported parameters for local OpenAI-compatible APIs."""
        # List of supported OpenAI chat completion parameters
        supported_params = {
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
            'presence_penalty', 'stop', 'stream', 'logit_bias', 'user',
            'seed', 'tools', 'tool_choice', 'response_format'
        }
        
        # Filter kwargs to only include supported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        # Log any filtered parameters for debugging
        filtered_out = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out:
            logger.debug(f"Filtered out unsupported parameters: {filtered_out}")
        
        return filtered_kwargs
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            filtered_kwargs = self._filter_kwargs(kwargs)
            messages = self._add_system_prompt_to_messages([{"role": "user", "content": prompt}])
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **filtered_kwargs
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in completion: {e}")
            raise
    
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            filtered_kwargs = self._filter_kwargs(kwargs)
            messages = self._add_system_prompt_to_messages([{"role": "user", "content": prompt}])
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **filtered_kwargs
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in async completion: {e}")
            raise
    
    @llm_completion_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponse:
        try:
            openai_messages = self._format_messages(messages)
            filtered_kwargs = self._filter_kwargs(kwargs)
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                **filtered_kwargs
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    @llm_completion_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponse:
        try:
            openai_messages = self._format_messages(messages)
            filtered_kwargs = self._filter_kwargs(kwargs)
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                **filtered_kwargs
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in async chat: {e}")
            raise
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        try:
            filtered_kwargs = self._filter_kwargs(kwargs)
            messages = self._add_system_prompt_to_messages([{"role": "user", "content": prompt}])
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                stream=True,
                **filtered_kwargs
            )
            
            def gen():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield CompletionResponse(
                            text=chunk.choices[0].delta.content, 
                            delta=chunk.choices[0].delta.content
                        )
            
            return gen()
        except Exception as e:
            logger.error(f"Error in stream completion: {e}")
            raise
    
    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs: Any):
        try:
            filtered_kwargs = self._filter_kwargs(kwargs)
            messages = self._add_system_prompt_to_messages([{"role": "user", "content": prompt}])
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                stream=True,
                **filtered_kwargs
            )
            
            async def gen():
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield CompletionResponse(
                            text=chunk.choices[0].delta.content, 
                            delta=chunk.choices[0].delta.content
                        )
            
            return gen()
        except Exception as e:
            logger.error(f"Error in async stream completion: {e}")
            raise
    
    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        try:
            openai_messages = self._format_messages(messages)
            filtered_kwargs = self._filter_kwargs(kwargs)
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                stream=True,
                **filtered_kwargs
            )
            
            def gen():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield CompletionResponse(
                            text=chunk.choices[0].delta.content, 
                            delta=chunk.choices[0].delta.content
                        )
            
            return gen()
        except Exception as e:
            logger.error(f"Error in stream chat: {e}")
            raise
    
    @llm_completion_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        try:
            openai_messages = self._format_messages(messages)
            filtered_kwargs = self._filter_kwargs(kwargs)
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                stream=True,
                **filtered_kwargs
            )
            
            async def gen():
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield CompletionResponse(
                            text=chunk.choices[0].delta.content, 
                            delta=chunk.choices[0].delta.content
                        )
            
            return gen()
        except Exception as e:
            logger.error(f"Error in async stream chat: {e}")
            raise
