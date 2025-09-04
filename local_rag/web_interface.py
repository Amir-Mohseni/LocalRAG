"""Gradio web interface for LocalRAG system."""

import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
import logging
import re
import os
import tempfile
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class LocalRAGChatInterface:
    """Chat interface for LocalRAG using Gradio."""
    
    def __init__(self, index: VectorStoreIndex):
        """
        Initialize the chat interface.
        
        Args:
            index: The vector store index for querying
        """
        self.index = index
        self.query_engine = index.as_query_engine()
        self.uploaded_files = []
        self.all_documents = []  # Store documents for incremental uploads
        logger.info("✅ Chat interface initialized")
    
    def chat_fn(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Process a chat message and return response with updated history.
        
        Args:
            message: User's input message
            history: Chat history as list of (user_msg, bot_msg) tuples
            
        Returns:
            Tuple of (empty_string, updated_history)
        """
        if not message.strip():
            return "", history
        
        try:
            # Query the RAG system
            response = self.query_engine.query(message)
            bot_response = str(response)
            
            # Filter out <think>...</think> tags if present
            bot_response = self._filter_thinking_tags(bot_response)
            
            # Update history
            history.append((message, bot_response))
            
            logger.info(f"Processed query: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            bot_response = f"Sorry, I encountered an error: {str(e)}"
            history.append((message, bot_response))
        
        return "", history
    
    def _filter_thinking_tags(self, response: str) -> str:
        """
        Filter out <think>...</think> tags from the response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            Cleaned response without thinking tags
        """
        # Remove <think>...</think> blocks (including multiline)
        pattern = r'<think>.*?</think>'
        cleaned_response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any extra whitespace
        cleaned_response = cleaned_response.strip()
        
        # If the response is empty after filtering, return a default message
        if not cleaned_response:
            cleaned_response = "I understand your question. Let me provide you with the relevant information from the documents."
        
        return cleaned_response
    
    def clear_chat(self) -> Tuple[List[Tuple[str, str]], str, str]:
        """Clear the chat history and reset everything."""
        self.uploaded_files = []
        self.all_documents = []
        
        try:
            from llama_index.core import Document
            empty_doc = Document(text="Welcome! Please upload documents to get started.")
            self.index = VectorStoreIndex.from_documents([empty_doc])
            self.query_engine = self.index.as_query_engine()
        except Exception as e:
            logger.warning(f"Could not reset to empty index: {e}")
        
        return [], None, "📄 No documents uploaded yet"
    
    def upload_documents(self, files) -> str:
        if not files:
            return "No files selected."
        
        try:
            new_documents = []
            new_file_names = []
            
            for file in files:
                file_path = file.name if hasattr(file, 'name') else file
                file_name = os.path.basename(file_path)
                
                if file_name in self.uploaded_files:
                    continue
                
                reader = SimpleDirectoryReader(input_files=[file_path])
                docs = reader.load_data()
                new_documents.extend(docs)
                new_file_names.append(file_name)
            
            if new_documents:
                # Add to stored documents
                self.all_documents.extend(new_documents)
                
                # Rebuild index with all stored documents
                self.index = VectorStoreIndex.from_documents(self.all_documents)
                self.query_engine = self.index.as_query_engine()
                self.uploaded_files.extend(new_file_names)
                
                total_docs = len(self.uploaded_files)
                logger.info(f"Added {len(new_documents)} new documents, total: {total_docs}")
                return f"✅ Added {len(new_documents)} new documents. Total: {total_docs} files loaded"
            else:
                return "No new documents to add (files may already be uploaded)"
                
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return f"❌ Error uploading documents: {str(e)}"
    
    def get_upload_status(self) -> str:
        """Get current upload status."""
        if self.uploaded_files:
            return f"📄 Currently loaded: {', '.join(self.uploaded_files)}"
        else:
            return "📄 No documents uploaded yet"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and return the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="LocalRAG Chat Interface",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 900px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 LocalRAG Chat Interface")
            
            # Document Upload Section
            file_upload = gr.File(
                label="📄 Add More Documents (Auto-indexing)",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".docx", ".md", ".csv"],
                height=120
            )
            
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=False,
                placeholder="Upload documents to get started..."
            )
            
            # Chat Section
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask a question about your documents...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    container=False,
                    scale=4
                )
                
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=80
                )
            
            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ Clear All & Reset",
                    variant="secondary"
                )
            
            # Event handlers
            def submit_message(message, history):
                return self.chat_fn(message, history)
            
            # Auto-upload when files are selected
            file_upload.upload(
                fn=self.upload_documents,
                inputs=[file_upload],
                outputs=[upload_status],
                queue=True
            )
            
            # Send button click
            send_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=True
            )
            
            # Enter key press
            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=True
            )
            
            # Clear button click - now clears everything
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, file_upload, upload_status],
                queue=False
            )
            
            # Initialize upload status
            interface.load(
                fn=self.get_upload_status,
                outputs=[upload_status],
                queue=False
            )
        
        return interface
    
    def launch(
        self, 
        share: bool = False, 
        server_name: str = "127.0.0.1", 
        server_port: int = 7860,
        debug: bool = False
    ):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
        """
        interface = self.create_interface()
        
        logger.info(f"🚀 Launching web interface at http://{server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True,
            quiet=False
        )


def create_web_interface(index: VectorStoreIndex) -> LocalRAGChatInterface:
    """
    Create a web interface for the LocalRAG system.
    
    Args:
        index: The vector store index
        
    Returns:
        LocalRAGChatInterface instance
    """
    return LocalRAGChatInterface(index)
