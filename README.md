# ChatBot RAG - AI-Powered Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot system that processes documents and provides intelligent answers using OpenAI's GPT-4o model with LangSmith tracing for observability.

## 📋 Requirements

- Python 3.11+
- Poetry for dependency management
- OpenAI API key
- ChromaDB (cloud or local)
- Optional: GPU support (CUDA)

## 🚀 Quick Start

### Local enviroment
1. Environment Setup

    - Clone the repository and navigate to the project:

        ```bash
        cd thesis
        ```

    - Create `.env` file with your configuration 
        > Note: I create an [.env.example](.env.example) file. Check it 


2. Install Dependencies

    ```bash
    poetry install
    ```
    
    ```bash
    poetry shell
    ```

3. Run the Application
    - Terminal 1 - API Server:
        ```bash
        poetry run main
        ```

    - Terminal 2 - Streamlit UI:
        ```bash
        poetry run ui
        ```

    > Note: The UI will be available at `http://localhost:8500`
 
    ---

    ## 🔧 Supported Document Formats

    - PDF: full docling PDF processing (OCR)
    - DOCX: optional; 
    - TXT/MD: plain text or markdown files are supported

    If you need additional formats (e.g., HTML, PPTX) we can add more converters or recommend additional dependencies.
---

### RUN in docker

- Build:
    ```bash
    docker build -t document-processor .
    ```

- Run:
    ```bash
    docker run --rm -it --gpus all -p 8000:8000 -p 8500:8500 document-processor:latest
    ```
> 8000 - port for API

> 8500 - port for UI
