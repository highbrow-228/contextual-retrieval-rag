# Contextual Retrieval for RAG

Code for the paper **"An Approach to Building RAG Systems Based on the Enhanced Contextual Retrieval Method"** (V. Koval, S. Krushelnytskyi), *Science and Technology Today*, No. 4(58), 2026.
DOI: [10.52058/2786-6025-2026-4(58)-3470-3481](https://doi.org/10.52058/2786-6025-2026-4(58)-3470-3481)

Standard RAG splits documents into chunks that lose their surrounding context ("semantic amnesia"): a chunk that says "revenue grew 3%" no longer says which company or which year. This project compares a **baseline RAG** with an **enhanced contextual retrieval** pipeline, where an LLM writes a short context description for every chunk before it is embedded.

## Results

FinanceBench, `novel-generated` questions (n = 50), evaluated with RAGAS (judge: gpt-4o-mini):

| Metric | Baseline | Contextual retrieval | Relative change |
|---|---|---|---|
| Context recall | 0.207 | 0.280 | **+35%** |
| Context precision | 0.280 | 0.387 | **+38%** |
| Answer correctness | 0.306 | 0.358 | +17% |
| Faithfulness | 0.614 | 0.662 | +8% |

The full evaluation run is in [`fill_database.ipynb`](fill_database.ipynb).

## How it works

```
Baseline:    document → Docling (OCR) → chunks → embeddings → ChromaDB → top-k → gpt-4o answer
Contextual:  document → Docling (OCR) → chunks → gpt-5-nano adds context to each chunk → embeddings → ChromaDB → top-k → gpt-4o answer
```

- **Both pipelines:** Docling document parsing with OCR, recursive chunking (1000 characters, 200 overlap), `text-embedding-3-large` embeddings, ChromaDB, top-k retrieval and a `gpt-4o` answer at temperature 0
- **Contextual retrieval** adds one step before embedding: `gpt-5-nano` writes a context description for each chunk, using the document name and a sliding window of the previous chunk summaries. Chunks are processed in concurrent batches of 30
- Baseline and contextual chunks live in **separate collections**, so both pipelines run side by side behind separate endpoints: `GET /chat/query_baseline` and `GET /chat/query_advanced`
- **Observability:** every request is traced in LangSmith
- **Local models:** an optional Ollama handler (Qwen3) for running without OpenAI

## Stack

Python 3.11, FastAPI, Streamlit, ChromaDB, OpenAI (gpt-4o, gpt-5-nano, text-embedding-3-large), Docling, RAGAS, LangSmith, Ollama, Docker, Poetry

## Quick start

### Requirements

- Python 3.11+
- Poetry
- OpenAI API key
- ChromaDB (cloud or local)
- Optional: GPU (CUDA)

### Local environment

```bash
cd contextual-retrieval-rag
cp .env.example .env   # fill in your keys
poetry install
```

Terminal 1, API server:

```bash
poetry run main
```

Terminal 2, Streamlit UI (http://localhost:8500):

```bash
poetry run ui
```

### Docker

```bash
docker build -t document-processor .
docker run --rm -it --gpus all -p 8000:8000 -p 8500:8500 document-processor:latest
```

- `8000`: API
- `8500`: UI

### Supported document formats

- **PDF:** full Docling processing with OCR
- **DOCX:** optional
- **TXT / MD:** plain text and Markdown

## Reproducing the evaluation

With the API running and the FinanceBench documents indexed into both collections, run [`fill_database.ipynb`](fill_database.ipynb). It sends the 50 `novel-generated` questions to `/chat/query_baseline` and `/chat/query_advanced` and scores both pipelines with RAGAS (`answer_correctness`, `faithfulness`, `context_recall`, `context_precision`).
