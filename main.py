#!/usr/bin/env python3
"""
LocalRAG - Main entry point for the local RAG system.

This script initializes and runs the LocalRAG system using configuration
from config.yaml. Supports both CLI and web interface modes.
"""

import sys
import os
import argparse
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_rag import create_local_rag_system, interactive_query_loop, create_web_interface

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LocalRAG - Local Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run CLI interface
  python main.py --web             # Run web interface
  python main.py --web --port 8080 # Run web interface on port 8080
  python main.py --web --share     # Run web interface with public sharing
        """
    )
    
    parser.add_argument(
        "--web", 
        action="store_true",
        help="Launch web interface instead of CLI"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    # Web interface options
    web_group = parser.add_argument_group("Web Interface Options")
    web_group.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for web interface (default: 127.0.0.1)"
    )
    
    web_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    web_group.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link for web interface"
    )
    
    web_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for web interface"
    )
    
    return parser.parse_args()


def run_cli_interface(index):
    """Run the CLI interface."""
    print("🖥️  Starting CLI interface...")
    print("Type 'exit' to quit.\n")
    interactive_query_loop(index)


def run_web_interface(index, args):
    """Run the web interface."""
    print("🌐 Starting web interface...")
    
    # Create web interface
    web_interface = create_web_interface(index)
    
    # Launch the interface
    web_interface.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug
    )


def main():
    """Main function to run the LocalRAG system."""
    args = parse_arguments()
    
    try:
        print("🚀 Initializing LocalRAG system...")
        
        # Create the RAG system using configuration
        index = create_local_rag_system(args.config)
        
        print("✅ LocalRAG system initialized successfully!")
        
        # Choose interface based on arguments
        if args.web:
            run_web_interface(index, args)
        else:
            run_cli_interface(index)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
