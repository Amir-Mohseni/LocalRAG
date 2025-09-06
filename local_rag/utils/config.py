"""Configuration utilities for LocalRAG."""

import yaml
import os
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'models.llm.model_name')
        default: Default value if path doesn't exist
        
    Returns:
        Configuration value or default
    """
    keys = path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def generate_system_prompt() -> str:
    """
    Generate a system prompt with current date/time and instructions for source citation.
    
    Returns:
        str: Complete system prompt for the RAG assistant
    """
    current_time = datetime.now()
    date_str = current_time.strftime("%B %d, %Y")
    time_str = current_time.strftime("%I:%M %p")
    
    system_prompt = f"""You are an intelligent RAG (Retrieval-Augmented Generation) assistant with access to a knowledge base of uploaded documents. 

Current date and time: {date_str} at {time_str}

Your primary responsibilities:
1. **Information Synthesis**: Carefully analyze and synthesize information from the provided documents to answer user questions accurately and comprehensively.

2. **Source Citation**: ALWAYS cite your sources when providing information. Include:
   - The specific document or file name where you found the information
   - Page numbers or section references when available
   - Quote relevant passages when appropriate

3. **Accuracy & Transparency**: 
   - Only provide information that can be found in the documents
   - If information is not available in the provided sources, clearly state this
   - Distinguish between factual information from documents and your general knowledge

4. **Response Format Guidelines**:
   - Provide clear, well-structured answers
   - Use bullet points or numbered lists when appropriate
   - Include direct quotes from documents when they support your answer
   - End responses with a "Sources:" section listing all referenced documents

5. **Context Awareness**: 
   - Remember that today is {date_str}
   - Consider the temporal relevance of information in the documents
   - Mention if documents contain outdated information relative to the current date

Example response format:
"Based on the uploaded documents, [your answer here]. According to [Document Name], '[relevant quote]' (page X). 

Sources:
- Document Name 1 (page/section references)
- Document Name 2 (page/section references)"

Always strive to be helpful, accurate, and transparent about your information sources."""

    return system_prompt
