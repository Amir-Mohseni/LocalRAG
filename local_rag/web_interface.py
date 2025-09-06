"""Gradio web interface for LocalRAG system."""

import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
import logging
import re
import os
import tempfile
from typing import List, Tuple, Optional
from .utils.model_manager import ModelManager
from .utils.config import generate_system_prompt
from .models.local_llm import LocalLLM

logger = logging.getLogger(__name__)


class LocalRAGChatInterface:
    """Chat interface for LocalRAG using Gradio."""
    
    def __init__(self, index: Optional[VectorStoreIndex]):
        self.index = index
        self.query_engine = index.as_query_engine() if index else None
        self.uploaded_files = []
        self.all_documents = []
        self.model_manager = ModelManager()
        if index:
            self._update_llm_system_prompt()
        logger.info("✅ Chat interface initialized")
    
    def initialize_rag_system(self):
        """Initialize the RAG system after models are configured."""
        try:
            from .rag_system import create_local_rag_system
            index = create_local_rag_system()
            if index:
                self.index = index
                self.query_engine = index.as_query_engine()
                self._update_llm_system_prompt()
                logger.info("✅ RAG system initialized after setup")
                return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
        return False
    
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
        
        if not self.query_engine:
            return "", history + [(message, "Please complete the model setup first before chatting.")]
        
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
    
    def _update_llm_system_prompt(self):
        """Update the LLM with a fresh system prompt including current date/time."""
        try:
            if isinstance(Settings.llm, LocalLLM):
                new_system_prompt = generate_system_prompt()
                Settings.llm._system_prompt = new_system_prompt
                logger.info("✅ Updated LLM system prompt with current date/time")
        except Exception as e:
            logger.warning(f"Could not update system prompt: {e}")
    
    def clear_chat(self) -> Tuple[List[Tuple[str, str]], None, str]:
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
                
                # Check if it's a directory
                if os.path.isdir(file_path):
                    reader = SimpleDirectoryReader(input_dir=file_path)
                    docs = reader.load_data()
                    new_documents.extend(docs)
                    new_file_names.append(f"{file_name}/ ({len(docs)} files)")
                else:
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    new_documents.extend(docs)
                    new_file_names.append(file_name)
            
            if new_documents:
                self.all_documents.extend(new_documents)
                self.index = VectorStoreIndex.from_documents(self.all_documents)
                self.query_engine = self.index.as_query_engine()
                self.uploaded_files.extend(new_file_names)
                
                # Update system prompt with fresh date/time
                self._update_llm_system_prompt()
                
                total_docs = len(self.uploaded_files)
                logger.info(f"Added {len(new_documents)} new documents, total: {total_docs}")
                return f"✅ Added {len(new_documents)} new documents. Total: {total_docs} items loaded"
            else:
                return "No new documents to add (files may already be uploaded)"
                
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return f"❌ Error uploading documents: {str(e)}"
    
    def index_folder(self, folder_path: str) -> str:
        """Index all documents in a folder."""
        if not folder_path or not os.path.exists(folder_path):
            return "Invalid folder path"
        
        try:
            reader = SimpleDirectoryReader(input_dir=folder_path)
            docs = reader.load_data()
            
            if docs:
                self.all_documents.extend(docs)
                self.index = VectorStoreIndex.from_documents(self.all_documents)
                self.query_engine = self.index.as_query_engine()
                
                # Update system prompt with fresh date/time
                self._update_llm_system_prompt()
                
                folder_name = os.path.basename(folder_path)
                self.uploaded_files.append(f"{folder_name}/ ({len(docs)} files)")
                
                logger.info(f"Indexed folder {folder_path} with {len(docs)} documents")
                return f"✅ Indexed {len(docs)} documents from {folder_name}"
            else:
                return "No documents found in folder"
                
        except Exception as e:
            logger.error(f"Error indexing folder: {e}")
            return f"❌ Error indexing folder: {str(e)}"
    
    def get_upload_status(self) -> str:
        if self.uploaded_files:
            return f"📄 Currently loaded: {', '.join(self.uploaded_files)}"
        else:
            return "📄 No documents uploaded yet"
    
    def get_file_list_html(self) -> str:
        """Generate HTML for the scrollable file list with remove buttons."""
        if not self.uploaded_files:
            return "<div style='text-align: center; color: #666; padding: 20px;'>No files uploaded yet</div>"
        
        html_items = []
        for i, file_name in enumerate(self.uploaded_files):
            # Escape HTML characters in file name
            safe_file_name = file_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_items.append(f"""
                <div class="file-item-row">
                    <div class="file-item-name">📄 {safe_file_name}</div>
                    <button class="file-remove-btn" onclick="window.removeFileByIndex({i})" 
                            style="background: #ff4757; color: white; border: none; border-radius: 4px; 
                                   padding: 4px 8px; cursor: pointer; font-size: 12px; 
                                   transition: background-color 0.2s;">❌</button>
                </div>
            """)
        
        return f"""
            <div>
                {''.join(html_items)}
            </div>
            <script>
                window.removeFileByIndex = function(index) {{
                    // Trigger the hidden Gradio button click
                    const hiddenBtn = document.getElementById('remove-file-btn-' + index);
                    if (hiddenBtn) {{
                        hiddenBtn.click();
                    }}
                }}
            </script>
        """
    
    def create_file_management_components(self):
        """Create dynamic components for file management"""
        components = []
        for i, file in enumerate(self.uploaded_files):
            with gr.Row():
                gr.Markdown(f"📄 `{file}`", elem_classes=["file-item"])
                remove_btn = gr.Button(
                    "❌",
                    variant="stop",
                    size="sm",
                    scale=0,
                    min_width=40
                )
                # Store the filename in the button's value for identification
                remove_btn.click(
                    fn=lambda f=file: self.remove_file_simple(f),
                    outputs=[],
                    queue=True
                )
                components.append((file, remove_btn))
        return components
    
    def remove_file_simple(self, filename: str):
        """Simple file removal without complex state management"""
        if filename in self.uploaded_files:
            file_index = self.uploaded_files.index(filename)
            self.uploaded_files.remove(filename)
            
            if file_index < len(self.all_documents):
                del self.all_documents[file_index]
            
            # Rebuild index
            if self.all_documents:
                self.index = VectorStoreIndex.from_documents(self.all_documents)
                self.query_engine = self.index.as_query_engine()
                
                # Update system prompt with fresh date/time
                self._update_llm_system_prompt()
            
            logger.info(f"Removed file: {filename}")
        return f"Removed {filename}"
    
    def remove_file(self, file_to_remove: str) -> Tuple[str, str]:
        if file_to_remove not in self.uploaded_files:
            return "File not found", self.get_file_list()
        
        try:
            # Find index of file to remove
            file_index = self.uploaded_files.index(file_to_remove)
            
            # Remove from tracking lists
            self.uploaded_files.remove(file_to_remove)
            
            # Remove corresponding documents (assuming one doc per file)
            if file_index < len(self.all_documents):
                del self.all_documents[file_index]
            
            # Rebuild index with remaining documents
            if self.all_documents:
                self.index = VectorStoreIndex.from_documents(self.all_documents)
                self.query_engine = self.index.as_query_engine()
                
                # Update system prompt with fresh date/time
                self._update_llm_system_prompt()
                
                status = f"✅ Removed {file_to_remove}. {len(self.uploaded_files)} files remaining"
            else:
                # Create empty index if no documents left
                from llama_index.core import Document
                empty_doc = Document(text="Upload documents to get started.")
                self.index = VectorStoreIndex.from_documents([empty_doc])
                self.query_engine = self.index.as_query_engine()
                
                # Update system prompt with fresh date/time
                self._update_llm_system_prompt()
                
                status = "✅ All files removed"
            
            logger.info(f"Removed file: {file_to_remove}")
            return status, self.get_file_list_html()
            
        except Exception as e:
            logger.error(f"Error removing file: {e}")
            return f"❌ Error removing file: {str(e)}", self.get_file_list_html()
    
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
            /* Global font styling - using a reliable font stack */
            *, body, .gradio-container, .gr-button, .gr-textbox, .gr-dropdown, .gr-markdown,
            h1, h2, h3, h4, h5, h6, p, span, div, label, .markdown {
                font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif !important;
            }
            
            /* Ensure consistent font rendering */
            * {
                -webkit-font-smoothing: antialiased !important;
                -moz-osx-font-smoothing: grayscale !important;
                text-rendering: optimizeLegibility !important;
            }
            
            .gradio-container {
                max-width: 1400px !important;
                margin: auto !important;
            }
            .chat-column {
                min-width: 500px !important;
                max-width: 900px !important;
                flex-shrink: 0 !important;
            }
            .upload-column {
                min-width: 350px !important;
                max-width: 450px !important;
                flex-shrink: 0 !important;
                margin-left: 20px !important;
            }
            .file-list-container {
                max-height: 300px !important;
                overflow-y: auto !important;
                border: 1px solid var(--border-color-primary) !important;
                border-radius: 8px !important;
                padding: 10px !important;
                background: var(--background-fill-secondary) !important;
                margin-top: 10px !important;
            }
            .file-item-row {
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                padding: 8px 12px !important;
                margin: 4px 0 !important;
                border: 1px solid var(--border-color-accent) !important;
                border-radius: 6px !important;
                background: var(--background-fill-primary) !important;
            }
            .file-item-name {
                flex: 1 !important;
                margin: 0 !important;
                padding: 0 !important;
                font-size: 14px !important;
                word-break: break-word !important;
            }
            .file-remove-btn {
                margin-left: 10px !important;
                flex-shrink: 0 !important;
            }
            .main-row {
                gap: 20px !important;
                align-items: flex-start !important;
            }
            """
        ) as interface:
            
            # Always check if models are properly loaded (even if setup was completed)
            setup_completed = self.model_manager.is_setup_completed()
            models_loaded = self.model_manager.are_models_loaded()
            
            # If setup is completed, also check if configured models are actually loaded
            models_valid = True
            if setup_completed:
                configured_models = self.model_manager.get_configured_models()
                loaded_models = self.model_manager.get_loaded_models()
                
                llm_valid = configured_models.get('llm') and configured_models['llm'] in loaded_models['llm']
                embedding_valid = configured_models.get('embedding') and configured_models['embedding'] in loaded_models['embedding']
                models_valid = llm_valid and embedding_valid
            
            # Show setup screen if setup not completed OR models not valid
            show_setup = not setup_completed or not models_valid
            
            if show_setup:
                # Initial Setup Screen
                with gr.Column(visible=True) as setup_screen:
                    gr.Markdown("# 🚀 LocalRAG Setup")
                    
                    if setup_completed and not models_valid:
                        gr.Markdown("⚠️ **Configured models are not loaded!**")
                        configured_models = self.model_manager.get_configured_models()
                        if configured_models.get('llm'):
                            gr.Markdown(f"**Configured LLM:** `{configured_models['llm']}`")
                        if configured_models.get('embedding'):
                            gr.Markdown(f"**Configured Embedding:** `{configured_models['embedding']}`")
                        gr.Markdown("Please load these models in **LM Studio's** Local Server tab.")
                    elif not models_loaded:
                        gr.Markdown("⚠️ **No models loaded in LM Studio!**")
                        gr.Markdown("Please load both LLM and embedding models in **LM Studio's** Local Server tab.")
                    else:
                        gr.Markdown("Please configure your models to get started:")
                    
                    available_models = self.model_manager.get_available_models()
                    loaded_models = self.model_manager.get_loaded_models()
                    
                    # Show model status
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📊 Model Status")
                            if loaded_models["llm"]:
                                gr.Markdown(f"✅ **Loaded LLM models:** {', '.join(loaded_models['llm'])}")
                            else:
                                gr.Markdown("❌ **No LLM models loaded**")
                            
                            if loaded_models["embedding"]:
                                gr.Markdown(f"✅ **Loaded embedding models:** {', '.join(loaded_models['embedding'])}")
                            else:
                                gr.Markdown("❌ **No embedding models loaded**")
                    
                    with gr.Row():
                        with gr.Column():
                            # Only show loaded models - no fallback to available models
                            llm_choices = loaded_models["llm"]
                            if not llm_choices:
                                llm_setup_dropdown = gr.Dropdown(
                                    label="LLM Model (None Loaded)",
                                    choices=["No loaded models"],
                                    interactive=False
                                )
                            else:
                                llm_setup_dropdown = gr.Dropdown(
                                    label="LLM Model (✅ Loaded)",
                                    choices=llm_choices,
                                    value=llm_choices[0],
                                    interactive=True
                                )
                        
                        with gr.Column():
                            # Only show loaded models - no fallback to available models
                            embedding_choices = loaded_models["embedding"]
                            if not embedding_choices:
                                embedding_setup_dropdown = gr.Dropdown(
                                    label="Embedding Model (None Loaded)",
                                    choices=["No loaded models"],
                                    interactive=False
                                )
                            else:
                                embedding_setup_dropdown = gr.Dropdown(
                                    label="Embedding Model (✅ Loaded)",
                                    choices=embedding_choices,
                                    value=embedding_choices[0],
                                    interactive=True
                                )
                    
                    with gr.Row():
                        # Only allow setup if both loaded models are available
                        setup_btn = gr.Button(
                            "Complete Setup",
                            variant="primary",
                            interactive=bool(loaded_models["llm"] and loaded_models["embedding"])
                        )
                        refresh_btn = gr.Button(
                            "🔄 Refresh Models",
                            variant="secondary"
                        )
                    
                    # Guidance for users
                    if setup_completed and not models_valid:
                        gr.Markdown("""
                        ### 📋 To load your configured models:
                        1. Open **LM Studio**
                        2. Go to the **Local Server** tab
                        3. Load the models shown above in both LLM and Embedding slots
                        4. Start the server if not already running
                        5. Click **🔄 Refresh Models** above to verify they're loaded
                        """)
                    elif not models_loaded:
                        gr.Markdown("""
                        ### 📋 To load models in **LM Studio**:
                        1. Open **LM Studio**
                        2. Go to the **Local Server** tab
                        3. Click **Select a model to load** for both LLM and Embedding slots
                        4. Choose models from your downloaded collection
                        5. Start the server if not already running
                        6. Click **🔄 Refresh Models** above to update the dropdowns
                        
                        ⚠️ **Note:** You can only proceed when both models are loaded and running in **LM Studio**.
                        """)
                    
                    setup_status = gr.Textbox(
                        label="Setup Status",
                        interactive=False,
                        visible=False
                    )
            
            # Main Interface
            with gr.Column(visible=not show_setup) as main_interface:
                gr.Markdown("# 🤖 LocalRAG Chat Interface")
                
                
                with gr.Row(elem_classes=["main-row"]):
                    # Left Column - Chat Section
                    with gr.Column(scale=3, elem_classes=["chat-column"]):
                        chatbot = gr.Chatbot(
                            label="💬 Chat",
                            height=600,
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
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear All & Reset",
                            variant="secondary",
                            size="sm"
                        )
                    
                    # Right Column - Document Management
                    with gr.Column(scale=2, elem_classes=["upload-column"]):
                        gr.Markdown("### 📄 Document Management")
                        
                        file_upload = gr.File(
                            label="Upload Documents or Folders",
                            file_count="multiple",
                            file_types=[".txt", ".pdf", ".docx", ".md", ".csv"],
                            height=120,
                            type="filepath"
                        )
                        
                        # Folder upload
                        folder_upload = gr.File(
                            label="📁 Upload Folder",
                            file_count="directory",
                            height=80
                        )
                        
                        upload_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=False,
                            placeholder="Upload documents to get started...",
                            lines=2
                        )
                        
                        gr.Markdown("#### 📋 Uploaded Files")
                        
                        # Scrollable file list container
                        with gr.Column(elem_classes=["file-list-container"]) as file_list_container:
                            file_list_html = gr.HTML(
                                value="<div style='text-align: center; color: #666; padding: 20px;'>No files uploaded yet</div>",
                                elem_classes=["file-list-content"]
                            )
                        
                        # Hidden remove buttons for file operations
                        remove_buttons = []
                        for i in range(20):  # Support up to 20 files
                            btn = gr.Button(
                                f"Remove file {i}",
                                visible=False,
                                elem_id=f"remove-file-btn-{i}",
                                elem_classes=["remove-file-btn-hidden"]
                            )
                            remove_buttons.append(btn)
            
            # Event handlers
            def submit_message(message, history):
                return self.chat_fn(message, history)
            
            def upload_and_update(files):
                status = self.upload_documents(files)
                file_list_html_content = self.get_file_list_html()
                return status, file_list_html_content, None  # Include cleared upload area
            
            def remove_file_by_index(index):
                if 0 <= index < len(self.uploaded_files):
                    file_name = self.uploaded_files[index]
                    self.remove_file_simple(file_name)
                    status = f"✅ Removed {file_name}"
                else:
                    status = "File not found"
                file_list_html_content = self.get_file_list_html()
                return status, file_list_html_content
            
            def clear_all():
                chat, upload_area, status = self.clear_chat()
                file_list_html_content = self.get_file_list_html()
                return chat, upload_area, status, file_list_html_content
            
            def download_model(model_name):
                if not model_name.strip():
                    return "Please enter a model name"
                return self.model_manager.download_model(model_name.strip())
            
            def index_folder_handler(folder_path_str):
                if not folder_path_str.strip():
                    return "Please enter a folder path", self.get_file_list_html()
                status = self.index_folder(folder_path_str.strip())
                file_list_html_content = self.get_file_list_html()
                return status, file_list_html_content
            
            def complete_setup(llm_model, embedding_model):
                # Validate that we have valid loaded models
                if (not llm_model or not embedding_model or 
                    llm_model == "No loaded models" or embedding_model == "No loaded models" or
                    llm_model == "No models available" or embedding_model == "No models available"):
                    return gr.Column(visible=True), gr.Column(visible=False)
                
                # Double-check that the selected models are actually loaded
                loaded_models = self.model_manager.get_loaded_models()
                if llm_model not in loaded_models["llm"] or embedding_model not in loaded_models["embedding"]:
                    return gr.Column(visible=True), gr.Column(visible=False)
                
                success = self.model_manager.save_model_config(llm_model, embedding_model)
                if success:
                    # Initialize the RAG system now that models are configured
                    rag_initialized = self.initialize_rag_system()
                    if rag_initialized:
                        return gr.Column(visible=False), gr.Column(visible=True)
                    else:
                        return gr.Column(visible=True), gr.Column(visible=False)
                else:
                    return gr.Column(visible=True), gr.Column(visible=False)
            
            def refresh_models():
                """Refresh the model lists and interface based on current LM Studio state."""
                # Get updated model information
                loaded_models = self.model_manager.get_loaded_models()
                
                # Update LLM dropdown - only show loaded models
                llm_choices = loaded_models["llm"]
                if not llm_choices:
                    llm_dropdown = gr.Dropdown(
                        label="LLM Model (None Loaded)",
                        choices=["No loaded models"],
                        interactive=False
                    )
                else:
                    llm_dropdown = gr.Dropdown(
                        label="LLM Model (✅ Loaded)",
                        choices=llm_choices,
                        value=llm_choices[0],
                        interactive=True
                    )
                
                # Update embedding dropdown - only show loaded models
                embedding_choices = loaded_models["embedding"]
                if not embedding_choices:
                    embedding_dropdown = gr.Dropdown(
                        label="Embedding Model (None Loaded)",
                        choices=["No loaded models"],
                        interactive=False
                    )
                else:
                    embedding_dropdown = gr.Dropdown(
                        label="Embedding Model (✅ Loaded)",
                        choices=embedding_choices,
                        value=embedding_choices[0],
                        interactive=True
                    )
                
                # Update setup button state - only allow if both models are loaded
                setup_button = gr.Button(
                    "Complete Setup",
                    variant="primary",
                    interactive=bool(loaded_models["llm"] and loaded_models["embedding"])
                )
                
                return llm_dropdown, embedding_dropdown, setup_button
            
            # Auto-upload when files are selected
            file_upload.upload(
                fn=upload_and_update,
                inputs=[file_upload],
                outputs=[upload_status, file_list_html, file_upload],
                queue=True
            )
            
            # Folder upload
            folder_upload.upload(
                fn=upload_and_update,
                inputs=[folder_upload],
                outputs=[upload_status, file_list_html, folder_upload],
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
            
            # Setup completion
            if show_setup:
                setup_btn.click(
                    fn=complete_setup,
                    inputs=[llm_setup_dropdown, embedding_setup_dropdown],
                    outputs=[setup_screen, main_interface],
                    queue=True
                )
                
                # Refresh button click
                refresh_btn.click(
                    fn=refresh_models,
                    outputs=[llm_setup_dropdown, embedding_setup_dropdown, setup_btn],
                    queue=True
                )
            
            # Bind remove buttons
            for i, btn in enumerate(remove_buttons):
                btn.click(
                    fn=lambda idx=i: remove_file_by_index(idx),
                    outputs=[upload_status, file_list_html],
                    queue=True
                )
            
            # Clear button click - now clears everything
            clear_btn.click(
                fn=clear_all,
                outputs=[chatbot, file_upload, upload_status, file_list_html],
                queue=False
            )
            
            # Initialize interface
            interface.load(
                fn=lambda: [self.get_upload_status(), self.get_file_list_html()],
                outputs=[upload_status, file_list_html],
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


def create_web_interface(index: Optional[VectorStoreIndex]) -> LocalRAGChatInterface:
    """
    Create a web interface for the LocalRAG system.
    
    Args:
        index: The vector store index
        
    Returns:
        LocalRAGChatInterface instance
    """
    return LocalRAGChatInterface(index)
