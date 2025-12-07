"""Main pipeline orchestrating the document processing workflow."""

# Standard library imports
import time
from typing import List

# Third-party imports
from fastapi import UploadFile
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.config.db_config import CollectionName
from chatbot.config.pipeline_config import PipelineMode
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
    def process(
        self, mode: PipelineMode, user_input: str, n_results: int = 5
    ) -> Chat:
        """
        This method is required by BaseProcessor but is not used in DocumentProcessingPipeline.
        Please use `process_document` or `process_batch_documents` instead.
        Returns a message indicating that direct string processing is not supported.
        """
        match mode:
            case PipelineMode.BASELINE:
                return self._process_baseline(user_input, n_results)
            case PipelineMode.ADVANCED:
                return self._process_advanced(user_input, n_results)
            case _:
                raise ValueError(f"Unsupported pipeline mode: {mode}")

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
        start_time = time.time()
        logger.opt(colors=True).info(
            f"Starting processing for document <green>{upload_file.filename}</green> "
            f"into collection '<green>{collection_name.value}</green>'"
        )

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

            chunks: List[Chunk] = await self.embedder.create_embedding_async(
                chunks_without_embeddings, upload_file
            )

            # Store chunks in vector database
            self.db_handler.process(
                collection_name=collection_name, data=chunks
            )

            logger.opt(colors=True).info(
                f"Document <green>{upload_file.filename}</green> processed successfully: "
                f"<green>{len(chunks)}</green> chunks added to collection <green>{collection_name.value}</green>"
                f" in <yellow>{time.time() - start_time}</yellow> seconds"
            )
            return chunks

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error processing document '<green>{upload_file.filename}</green>' for collection '<green>{collection_name.value}</green>': <red>{str(e)}</red>"
            )
            raise

    @traceable(
        run_type="tool",
        name="Process Batch Documents",
        metadata={
            "type": "sequential",
            "ls_provider": "DocumentProcessingPipeline",
        },
    )
    async def process_batch_documents(
        self, upload_files: List[UploadFile], collection_name: CollectionName
    ) -> List[List[Chunk]]:
        """Process multiple uploaded documents and add chunks to the vector database."""
        batch_chunks: List[List[Chunk]] = []
        successful_docs = 0
        failed_docs = 0
        failed_doc_errors: dict[str, str] = {}

        for upload_file in upload_files:
            try:
                chunks = await self.process_document(
                    upload_file, collection_name
                )
                batch_chunks.append(chunks)
                successful_docs += 1
            except Exception as e:
                failed_docs += 1
                failed_doc_errors[upload_file.filename] = str(e)
                logger.opt(colors=True).warning(
                    f"Document <yellow>{upload_file.filename}</yellow> failed to process:"
                    f"<red>{e}</red>"
                )

        # Log batch processing statistics
        logger.opt(colors=True).info(
            f"Batch processing completed for collection '<green>{collection_name.value}</green>': "
            f"<green>{successful_docs}</green> successful, <red>{failed_docs}</red> failed"
        )

        if failed_docs > 0:
            logger.opt(colors=True).warning(
                f"Failed documents: {', '.join(failed_doc_errors.keys())}"
            )
            for filename, error in failed_doc_errors.items():
                logger.opt(colors=True).debug(
                    f"  - <yellow>{filename}</yellow>: <red>{error}</red>"
                )

        return batch_chunks

    @traceable(
        run_type="tool",
        name="Process Baseline",
    )
    def _process_baseline(self, user_input: str, n_results: int) -> Chat:
        """Process user input in baseline mode."""
        collection_name: CollectionName = CollectionName.BASELINE_DOCUMENTS

        try:
            logger.opt(colors=True).info(
                f"Received user input for RAG processing: <yellow>{user_input}</yellow>"
            )

            rag_results: RAGQueryResult = self.db_handler.retrieve(
                query=user_input,
                collection_name=collection_name,
                n_results=n_results,
            )

            if rag_results.retrieved_chunks == []:
                logger.opt(colors=True).warning(
                    f"No relevant chunks found in the database for input '<yellow>{user_input[:100]}</yellow>'."
                )
                return Chat(
                    llm_response=(
                        "I'm sorry, but I couldn't find any relevant information in the documents to answer your question.\n"
                        "No relevant chunks were found in the database."
                    )
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
        name="Process Advanced",
    )
    def _process_advanced(self, user_input: str, n_results: int) -> Chat:
        collection_name: CollectionName = CollectionName.ADVANCED_DOCUMENTS
        # TODO: Implement advanced processing logic
        try:
            logger.opt(colors=True).info(
                f"Received user input for RAG processing: <yellow>{user_input}</yellow>"
            )

            rag_results: RAGQueryResult = self.db_handler.retrieve(
                query=user_input,
                collection_name=collection_name,
                n_results=n_results,
            )

            if rag_results.retrieved_chunks == []:
                logger.opt(colors=True).warning(
                    f"No relevant chunks found in the database for input '<yellow>{user_input[:100]}</yellow>'."
                )
                return Chat(
                    llm_response=(
                        "I'm sorry, but I couldn't find any relevant information in the documents to answer your question.\n"
                        "No relevant chunks were found in the database."
                    )
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
            batch_size=openai_settings.MAX_CONCURRENT_REQUESTS_EMBEDDING,
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
            concurent_requests_for_upload=db_settings.CONCURENT_DB_REQUESTS_FOR_UPLOAD,
        )

    @log_initialization("Document Processor")
    def _initialize_document_processor(self) -> None:
        self.document_processor: Processor = Processor(
            engine=ocr_settings.ENGINE,
        )

    @log_initialization("Chunker")
    def _initialize_chunker(self) -> None:
        self.chunker: Chunker = Chunker(
            chunk_size=chunk_settings.CHUNK_SIZE,
            chunk_overlap=chunk_settings.CHUNK_OVERLAP,
        )
