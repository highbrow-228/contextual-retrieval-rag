"""Main pipeline orchestrating the document processing workflow."""

# Standard library imports
from typing import List

# Third-party imports
from fastapi import UploadFile
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.config.db_config import CollectionName
from chatbot.globals import (
    chunk_settings,
    db_settings,
    ocr_settings,
    openai_settings,
)
from chatbot.models.chat import Chat
from chatbot.models.chunk import Chunk
from chatbot.models.rag import RAGQueryResult
from chatbot.services.chunking.chunker import Chunker
from chatbot.services.document_processing.processor import Processor
from chatbot.services.embedding.openai_embedder import OpenAIEmbedder
from chatbot.services.llm_handlers.openai_handler import OpenAIHandler
from chatbot.utils.log_utils import log_initialization
from chatbot.vector_db.vector_db_client import ChromaDBClient


class DocumentProcessingPipeline(BaseProcessor):
    """Pipeline for processing documents from PDF to vector database."""

    def __init__(self) -> None:
        self._initialize_openai_handler()
        self._initialize_embedder()
        self._initialize_db_handler()
        self._initialize_document_processor()
        self._initialize_chunker()

    @traceable(
        name="RAG",
        metadata={
            "type": "sequential",
            "ls_provider": "DocumentProcessingPipeline",
        },
    )
    def process(self, user_input: str, n_results: int = 5) -> Chat:
        """
        This method is required by BaseProcessor but is not used in DocumentProcessingPipeline.
        Please use `process_document` or `process_batch_documents` instead.
        Returns a message indicating that direct string processing is not supported.
        """
        try:
            logger.opt(colors=True).info(
                f"Received user input for RAG processing: <yellow>{user_input}</yellow>"
            )

            rag_results: RAGQueryResult = self.db_handler.retrieve(
                query=user_input,
                collection_name=CollectionName.RAG_DOCUMENT,
                n_results=n_results,
            )

            text_chunks: str = self.chunker.format_retrieved_chunks(
                rag_results.retrieved_chunks
            )

            # Generate LLM response using RAG
            llm_response: str = self.openai_handler.invoke_rag(
                user_input=user_input,
                text_chunks=text_chunks,
            )

            return Chat(llm_response=llm_response)

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error during RAG processing for input '<yellow>{user_input[:100]}</yellow>': <red>{str(e)}</red>"
            )
            raise

    @traceable(
        run_type="tool",
        name="Process Document",
        metadata={
            "type": "sequential",
            "ls_provider": "DocumentProcessingPipeline",
        },
    )
    async def process_document(
        self, upload_file: UploadFile, collection_name: CollectionName
    ) -> List[Chunk]:
        """Process a single uploaded document and add chunks to the vector database."""
        try:
            # Extract text from uploaded document
            text: str = await self.document_processor.process(upload_file)

            # Chunk the extracted text
            chunks_without_embeddings: List[Chunk] = (
                await self.chunker.process(
                    text_to_split=text,
                    filename=upload_file.filename,
                )
            )

            chunks: List[Chunk] = self.embedder.create_embedding(
                chunks_without_embeddings, upload_file
            )

            # Store chunks in vector database
            self.db_handler.process(
                collection_name=collection_name, data=chunks
            )

            return chunks

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error processing document '<green>{upload_file.filename}</green>' for collection '<green>{collection_name.value}</green>': <red>{str(e)}</red>"
            )
            raise

    async def process_batch_documents(
        self, upload_files: List[UploadFile], collection_name: CollectionName
    ) -> List[List[Chunk]]:
        """Process multiple uploaded documents and add chunks to the vector database."""
        try:
            # Process documents to extract text
            documents: list[tuple[str, str]] = (
                await self.document_processor.process_batch(upload_files)
            )

            # Process chunking in batch
            batch_chunks: List[List[Chunk]] = await self.chunker.process_batch(
                documents
            )

            # Embed all chunks from the batch
            total_chunks = 0
            for doc_idx, chunks_list in enumerate(batch_chunks):
                try:
                    filename = (
                        upload_files[doc_idx].filename
                        if doc_idx < len(upload_files)
                        else f"Document {doc_idx}"
                    )
                    for chunk_idx, chunk in enumerate(chunks_list):
                        try:
                            embeddings: list[float] = self.embedder.process(
                                chunk.content
                            )
                            chunk.embeddings = embeddings
                        except Exception as e:
                            logger.opt(colors=True).error(
                                f"Error embedding chunk <yellow>{chunk_idx}</yellow> from <green>{filename}</green>: <red>{str(e)}</red>"
                            )
                            raise
                    total_chunks += len(chunks_list)
                except Exception as e:
                    logger.opt(colors=True).error(
                        f"Error processing document <yellow>{doc_idx}</yellow> in batch: <red>{str(e)}</red>"
                    )
                    raise

            # Store all chunks in vector database
            for chunks_list in batch_chunks:
                self.db_handler.process(
                    collection_name=collection_name, data=chunks_list
                )

            return batch_chunks

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error processing batch of <yellow>{len(upload_files)}</yellow> documents for collection '<green>{collection_name.value}</green>': <red>{str(e)}</red>"
            )
            raise

    ### Private initialization methods ###
    @log_initialization("OpenAI Handler")
    def _initialize_openai_handler(self) -> None:
        self.openai_handler: OpenAIHandler = OpenAIHandler(
            api_key=openai_settings.OPENAI_API_KEY,
            rag_model_name=openai_settings.MODEL_NAME_RAG,
            rag_temperature=openai_settings.TEMPERATURE_RAG,
            rag_max_tokens=openai_settings.MAX_TOKENS_RAG,
            rag_max_retries=openai_settings.MAX_RETRIES_INVOKE_LLM,
            base_delay=openai_settings.BASE_DELAY_RETRY_LLM,
        )

    @log_initialization("OpenAI Embedder")
    def _initialize_embedder(self) -> None:
        self.embedder: OpenAIEmbedder = OpenAIEmbedder(
            api_key=openai_settings.OPENAI_API_KEY,
            embedding_model_name=openai_settings.MODEL_NAME_EMBEDDING,
        )

    @log_initialization("ChromaDB client")
    def _initialize_db_handler(self) -> None:
        self.db_handler: ChromaDBClient = ChromaDBClient(
            embedder=self.embedder,
            mode=db_settings.CHROMA_MODE,
            host=db_settings.CHROMA_HOST,
            port=db_settings.CHROMA_PORT,
            tenant=db_settings.CHROMA_TENANT,
            api_key=db_settings.CHROMA_API_KEY,
        )

    @log_initialization("Document Processor")
    def _initialize_document_processor(self) -> None:
        self.document_processor: Processor = Processor(
            engine=ocr_settings.ENGINE,
            code_enrichment=ocr_settings.CODE_ENRICHMENT,
            formula_enrichment=ocr_settings.FORMULA_ENRICHMENT,
            num_threads=ocr_settings.OMP_NUM_THREADS,
        )

    @log_initialization("Chunker")
    def _initialize_chunker(self) -> None:
        self.chunker: Chunker = Chunker(
            chunk_size=chunk_settings.CHUNK_SIZE,
            chunk_overlap=chunk_settings.CHUNK_OVERLAP,
        )
