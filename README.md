# RAG Application with LangChain

This is a Retrieval-Augmented Generation (RAG) application built using LangChain. It processes Python (.py) and Markdown (.md) files from a specified directory and allows you to ask questions about their content.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required system packages for document processing:
  - tesseract-ocr (for OCR capabilities)
  - libmagic (for file type detection)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place the Python (.py) and/or Markdown (.md) files you want to process in the project directory or its subdirectories.

2. Run the RAG application:
```bash
python rag_app.py
```

3. The application will:
   - Load and process all Python and Markdown files in the directory
   - Create a vector store using Chroma
   - Start an interactive question-answering session

4. Type your questions and press Enter. The application will:
   - Search for relevant information in the processed documents
   - Generate an answer using the OpenAI model
   - Show the source documents used for the answer

5. Type 'quit' to exit the application.

## Features

- Supports Python (.py) and Markdown (.md) files
- Maintains conversation history for context-aware responses
- Uses ChromaDB for efficient vector storage
- Provides source attribution for answers
- Multi-threaded document processing

## Note

Make sure you have sufficient OpenAI API credits, as the application uses the API for:
- Generating embeddings for document chunks
- Creating responses to questions 