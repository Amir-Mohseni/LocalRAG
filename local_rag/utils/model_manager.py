"""Model management utilities for LM Studio integration."""

import logging
import yaml
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

import lmstudio as lms


class ModelManager:
    """Manage LM Studio models and configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
    
    def is_setup_completed(self) -> bool:
        """Check if initial setup is completed."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('setup', {}).get('completed', False)
        except:
            return False
    
    def save_model_config(self, llm_model: str, embedding_model: str):
        """Save selected models to config and mark setup as completed."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['models']['llm']['model_name'] = llm_model
            config['models']['embedding']['model_name'] = embedding_model
            config['setup']['completed'] = True
            
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            
            logger.info(f"Saved model configuration: LLM={llm_model}, Embedding={embedding_model}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get_configured_models(self) -> Dict[str, str]:
        """Get currently configured models."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {
                    'llm': config['models']['llm']['model_name'],
                    'embedding': config['models']['embedding']['model_name']
                }
        except:
            return {'llm': '', 'embedding': ''}
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get available models from LM Studio (both downloaded and loaded)."""
        try:
            # Get by category
            downloaded_llm = lms.list_downloaded_models("llm")
            downloaded_embedding = lms.list_downloaded_models("embedding")
            loaded_llm = lms.list_loaded_models("llm")
            loaded_embedding = lms.list_loaded_models("embedding")
            
            # Extract model identifiers properly
            def extract_model_identifier(model):
                # Try different attributes to get the clean model identifier
                if hasattr(model, 'identifier'):
                    return model.identifier
                elif hasattr(model, 'model_key'):
                    return model.model_key
                elif hasattr(model, 'model_id'):
                    return model.model_id
                elif hasattr(model, 'name'):
                    return model.name
                elif hasattr(model, 'display_name'):
                    return model.display_name
                else:
                    # If it's a string representation like "LLM(identifier='qwen/qwen3-4b-thinking-2507')"
                    # extract the identifier value
                    model_str = str(model)
                    if "identifier='" in model_str:
                        start = model_str.find("identifier='") + len("identifier='")
                        end = model_str.find("'", start)
                        if end > start:
                            return model_str[start:end]
                    elif "identifier=\"" in model_str:
                        start = model_str.find("identifier=\"") + len("identifier=\"")
                        end = model_str.find("\"", start)
                        if end > start:
                            return model_str[start:end]
                    return model_str
            
            # Combine and deduplicate using proper model identifiers
            llm_names = []
            for model in downloaded_llm:
                llm_names.append(extract_model_identifier(model))
            for model in loaded_llm:
                llm_names.append(extract_model_identifier(model))
            
            embedding_names = []
            for model in downloaded_embedding:
                embedding_names.append(extract_model_identifier(model))
            for model in loaded_embedding:
                embedding_names.append(extract_model_identifier(model))
            
            # Remove duplicates while preserving order
            all_llm = list(dict.fromkeys(llm_names))
            all_embedding = list(dict.fromkeys(embedding_names))
            
            logger.info(f"Found LLM models: {all_llm}")
            logger.info(f"Found embedding models: {all_embedding}")
            
            return {
                "llm": all_llm,
                "embedding": all_embedding
            }
        except Exception as e:
            logger.error(f"Error getting LM Studio models: {e}")
            return {"llm": [], "embedding": []}
    
    @staticmethod
    def get_loaded_models() -> Dict[str, List[str]]:
        """Get only loaded models from LM Studio."""
        try:
            loaded_llm = lms.list_loaded_models("llm")
            loaded_embedding = lms.list_loaded_models("embedding")
            
            # Extract model identifiers properly
            def extract_model_identifier(model):
                # Try different attributes to get the clean model identifier
                if hasattr(model, 'identifier'):
                    return model.identifier
                elif hasattr(model, 'model_key'):
                    return model.model_key
                elif hasattr(model, 'model_id'):
                    return model.model_id
                elif hasattr(model, 'name'):
                    return model.name
                elif hasattr(model, 'display_name'):
                    return model.display_name
                else:
                    # If it's a string representation like "LLM(identifier='qwen/qwen3-4b-thinking-2507')"
                    # extract the identifier value
                    model_str = str(model)
                    if "identifier='" in model_str:
                        start = model_str.find("identifier='") + len("identifier='")
                        end = model_str.find("'", start)
                        if end > start:
                            return model_str[start:end]
                    elif "identifier=\"" in model_str:
                        start = model_str.find("identifier=\"") + len("identifier=\"")
                        end = model_str.find("\"", start)
                        if end > start:
                            return model_str[start:end]
                    return model_str
            
            llm_names = [extract_model_identifier(model) for model in loaded_llm]
            embedding_names = [extract_model_identifier(model) for model in loaded_embedding]
            
            logger.info(f"Found loaded LLM models: {llm_names}")
            logger.info(f"Found loaded embedding models: {embedding_names}")
            
            return {
                "llm": llm_names,
                "embedding": embedding_names
            }
        except Exception as e:
            logger.error(f"Error getting loaded LM Studio models: {e}")
            return {"llm": [], "embedding": []}
    
    @staticmethod
    def are_models_loaded() -> bool:
        """Check if both LLM and embedding models are loaded in LM Studio."""
        loaded_models = ModelManager.get_loaded_models()
        return len(loaded_models["llm"]) > 0 and len(loaded_models["embedding"]) > 0
    
    @staticmethod
    def get_fallback_models() -> Dict[str, List[str]]:
        """Get fallback model names for manual configuration."""
        return {
            "llm": [
                "qwen/qwen3-4b-thinking-2507",
                "llama-3.2-1b-instruct",
                "llama-3.2-3b-instruct",
            ],
            "embedding": [
                "text-embedding-embeddinggemma-300m-qat",
                "nomic-ai/nomic-embed-text-v1.5",
            ]
        }
