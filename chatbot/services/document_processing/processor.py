"""PDF processing module for converting PDF documents to structured text."""

# Standard library imports
import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

# Third-party imports
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    LayoutOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types import DoclingDocument
from docx import Document as DocxDocument
from fastapi import UploadFile
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.base.base_classes import BaseProcessor


# Disable verbose logging from docling libraries
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling_core").setLevel(logging.WARNING)


class Processor(BaseProcessor):
    """Handles PDF file operations."""

    # Supported formats that this processor implements. Keep in sync with
    # the per-format handlers: `_process_pdf`, `_process_txt`, `_process_md`, `_process_docx`.
    SUPPORTED_FORMATS: ClassVar[List[str]] = ["pdf", "txt", "md", "docx"]

    def __init__(self, engine: str) -> None:
        self.engine = engine

        self.__init_pdf_pipeline_options()

    @traceable(
        run_type="chain",
        name="Process Upload File",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    async def process(self, upload_file: UploadFile) -> Optional[str]:
        """
        Export an uploaded document to plain text (string).

        Supported formats:
        - **pdf**: Uses docling PDF pipeline to extract structured text (markdown).
        - **txt**, **md**: Read file content as plain text.
        - **docx**: Attempt to use `python-docx` to extract text; if not available, instructs to install it.

        Parameters
        ----------
        - **upload_file** (UploadFile): The uploaded file to process.

        Returns
        -------
        - **str | None**: The processed textual content (e.g., markdown) or None where not applicable.

        Raises
        ------
        - **ValueError**: If the file format is unsupported or necessary libraries are missing.
        """
        suffix = Path(upload_file.filename).suffix.lower().replace(".", "")

        logger.opt(colors=True).info(
            f"Processing uploaded file: <green>{upload_file.filename}</green>"
        )
        logger.opt(colors=True).debug(
            f"Detected file format: <green>{suffix}</green> for file: <green>{upload_file.filename}</green>"
        )

        # Create a temporary directory for temporary files
        TEMP_DIR = Path("temp_pdfs")
        TEMP_DIR.mkdir(exist_ok=True)
        tmp_path = None

        try:
            # Save temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{suffix}", dir=TEMP_DIR
            ) as tmp:
                tmp.write(await upload_file.read())
                tmp_path = Path(tmp.name)
                logger.opt(colors=True).debug(
                    f"Saved temporary <green>{upload_file.filename}</green> at: <green>{tmp_path}</green>"
                )

            suffix_lower = suffix.lower()

            match suffix_lower:
                case "pdf":
                    # Use the Docling PDF pipeline for PDF files
                    return self._process_pdf(tmp_path)
                case "txt":
                    return self._process_txt(tmp_path)
                case "md":
                    return self._process_md(tmp_path)
                case "docx":
                    return self._process_docx(tmp_path)
                case _:
                    raise ValueError(
                        f"Unsupported file type: {suffix.lower()}. Supported types: [{', '.join(self.SUPPORTED_FORMATS)}]"
                    )
        finally:
            # Clean up temporary file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
                logger.opt(colors=True).debug(
                    f"Deleted temporary file: <green>{tmp_path}</green>"
                )

    @traceable(
        run_type="tool",
        name="Process Batch",
        metadata={
            "type": "Batch",
            "ls_provider": "DoclingProcessor",
        },
    )
    async def process_batch(
        self, upload_files: List[UploadFile]
    ) -> List[Tuple[Optional[str], str]]:
        tasks = [
            self.process(upload_file)
            for upload_file in upload_files
        ]

        return await asyncio.gather(*tasks)

    @traceable(
        run_type="tool",
        name="Process PDF file",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    def _process_pdf(self, tmp_path: Path) -> str:
        """Convert a PDF to a DoclingDocument using configured Docling pipeline."""
        start_time = time.time()

        converter: DocumentConverter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pdf_pipeline_options,
                )
            },
        )

        result: ConversionResult = converter.convert(tmp_path)

        doc: DoclingDocument = result.document

        text: str = doc.export_to_text()

        logger.opt(colors=True).info(
            f"Completed PDF processing in <yellow>{time.time() - start_time:.2f}</yellow> seconds for temp file: <green>{tmp_path.name}</green>"
        )

        return text

    @traceable(
        run_type="tool",
        name="Process TXT file",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    def _process_txt(self, tmp_path: Path) -> str:
        """Process plain text files and return content (prefer Docling markdown if available)."""
        # Prefer Docling normalization but fall back to raw read if unavailable
        try:
            return self._process_with_docling(tmp_path, InputFormat.TXT)
        except Exception:
            # Minimised try/except here — if docling fails, fall back to straightforward read
            try:
                with open(tmp_path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except Exception:
                raise ValueError("Failed to process TXT: unable to read file.")

    @traceable(
        run_type="tool",
        name="Process MD file",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    def _process_md(self, tmp_path: Path) -> str:
        """Process markdown files using Docling to extract structure."""
        try:
            start_time = time.time()

            converter = DocumentConverter()

            result: ConversionResult = converter.convert(tmp_path)

            doc: DoclingDocument = result.document

            text: str = doc.export_to_text()

            logger.opt(colors=True).info(
                f"Completed MD processing in {time.time() - start_time:.2f} seconds for temp file: <green>{tmp_path.name}</green>"
            )

            return text

        except Exception as e:

            raise ValueError(f"Failed to process MD: {str(e)}")

    @traceable(
        run_type="tool",
        name="Process DOCX file",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    def _process_docx(self, tmp_path: Path) -> str:
        """Process docx files; try docling first and fallback to python-docx parsing."""
        try:
            return self._process_with_docling(tmp_path, InputFormat.DOCX)
        except Exception:
            # Single fallback try/except to python-docx parsing
            try:
                doc = DocxDocument(tmp_path)
                paragraphs: List[str] = [
                    p.text for p in doc.paragraphs if p.text.strip()
                ]
                return "\n\n".join(paragraphs)
            except Exception as e:
                raise ValueError(f"Failed to process DOCX: {str(e)}")

    @traceable(
        run_type="tool",
        name="Process with Filename",
        metadata={
            "type": "sequential",
            "ls_provider": "DoclingProcessor",
        },
    )
    async def _process_with_filename(
        self, upload_file: UploadFile
    ) -> Tuple[Optional[str], str]:
        """Process a file and return tuple of (text, filename)."""
        text = await self.process(upload_file)
        return (text, upload_file.filename)

    @traceable(
        run_type="tool",
        name="Docling Format Converter",
        metadata={
            "type": "internal",
            "ls_provider": "DoclingProcessor",
        },
    )
    def _process_with_docling(
        self, tmp_path: Path, input_format: InputFormat
    ) -> str:
        """Try to convert a file using Docling DocumentConverter and return markdown.

        This is used to keep TXT/MD/DOCX handling consistent with PDF output,
        returning markdown-like structured text when possible.
        """
        start_time = time.time()

        converter: DocumentConverter = DocumentConverter(
            allowed_formats=[input_format],
        )

        result: ConversionResult = converter.convert(tmp_path)

        doc: DoclingDocument = result.document

        text: str = doc.export_to_text()

        logger.opt(colors=True).info(
            f"Completed {input_format.value} processing in {time.time() - start_time:.2f} seconds for temp file: <green>{tmp_path.name}</green>"
        )

        return text

    def __init_pdf_pipeline_options(self) -> None:
        """Initialize PDF processor with Docling settings."""
        self.pdf_pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=os.cpu_count(),  # Use all available CPU cores for parallel processing
                device=AcceleratorDevice.CUDA,
            ),
            # Pictures
            do_picture_classification=True,  # Classify and categorize images
            do_picture_description=True,  # Generate descriptions for images
            images_scale=5.0,  # Upscale images 5x for better OCR quality
            # Tables
            do_table_structure=True,  # Extract table structure and layout
            do_table_enrichment=True,  # Enhance table data extraction
            do_code_enrichment=True,  # Extract code from documents
            do_cell_matching=True,  # Match table cells accurately
            do_formula_enrichment=True,  # Extract mathematical formulas
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,  # Align table cells precisely
                mode=TableFormerMode.ACCURATE,  # Use high-accuracy table parsing mode
            ),
            # OCR
            do_ocr=True,  # Enable optical character recognition
            ocr_options=EasyOcrOptions(
                lang=["en", "uk", "ru"],
                model_storage_directory="ocr_models",  # Path to store downloaded OCR models
                confidence_threshold=0.3,  # Lower value → detect more uncertain text
                bitmap_area_threshold=0.001,  # Lower value → capture smaller text regions
                download_enabled=True,  # Auto-download missing OCR models
                suppress_mps_warnings=True,  # Hide irrelevant MPS (Metal) warnings
            ),
            # Layout
            enable_page_segmentation=True,  # Divide pages into logical sections
            preserve_visual_order=True,  # Keep original reading order
            layout_options=LayoutOptions(
                create_orphan_clusters=True,  # Group isolated text elements
            ),
        )
