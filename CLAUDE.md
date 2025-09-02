# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system for course content, built with FastAPI backend and vanilla frontend. The system processes structured course documents and provides intelligent search capabilities using Zhipu AI's GLM-4.5 model and ChromaDB for vector storage.

## Development Setup

### Environment Setup
- Install dependencies: `uv sync`
- Set environment variable: `ZHIPUAI_API_KEY` (required for Zhipu AI API)
- The system uses Python 3.13+ with uv as package manager

### Running the Application
- Development server: `./run.sh` or `uv run uvicorn app:app --reload --port 8000`
- Web interface: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

## Architecture Overview

### Core Components
- **RAG System** (`backend/rag_system.py`) - Main orchestrator coordinating all components
- **Vector Store** (`backend/vector_store.py`) - ChromaDB wrapper with separate collections for course metadata and content
- **Document Processor** (`backend/document_processor.py`) - Parses structured course documents with specific format requirements
- **AI Generator** (`backend/ai_generator.py`) - Handles Zhipu AI API interactions with tool calling
- **Search Tools** (`backend/search_tools.py`) - Implements tool-based search pattern for AI
- **Session Manager** (`backend/session_manager.py`) - Manages conversation history with configurable limits

### Key Design Patterns
- **Tool-Based Search**: AI uses structured tools (`get_courses`, `get_lessons`, `get_lesson_content`) for course content queries
- **Separation of Concerns**: Clear module boundaries with dependency injection
- **Semantic Search**: Smart course name matching via vector embeddings
- **Dual Collections**: ChromaDB maintains separate collections for course metadata and lesson content

## Document Processing Requirements

The system expects structured course documents with specific formatting:
- First 3 lines: Course metadata (title, link, instructor)
- Lesson format: "Lesson X: [title]" pattern
- Content chunked with configurable size (800 chars) and overlap (100 chars)
- Documents stored in `docs/` directory

## Database Storage

- **Vector Database**: ChromaDB with persistent storage in `./chroma_db/`
- **Collections**: Separate collections for courses metadata and lesson content
- **Embeddings**: Sentence transformers for semantic search capabilities

## API Structure

- **POST /query** - Main chat endpoint for course content queries
- **GET /** - Web interface serving static files
- **WebSocket support** for real-time interactions

## Testing & Development

The system includes comprehensive error handling and logging. When making changes:
- Test document processing with new course materials
- Verify search functionality through the web interface
- Check API responses using the interactive docs at `/docs`
- Monitor ChromaDB storage in `./chroma_db/` directory

## Configuration

Environment variables:
- `ZHIPUAI_API_KEY` - Required for AI functionality
- Port and host settings configured in `app.py`

The system uses dependency injection patterns, making it easy to swap components or add new data sources.