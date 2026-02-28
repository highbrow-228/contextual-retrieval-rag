# Third-party imports
from langchain_ollama import ChatOllama
from loguru import logger
from ollama import Client


class OllamaHandler:
    """Handles interactions with the Ollama LLM provider."""

    def __init__(self) -> None:
        # self.client = ChatOllama(
        #     # base_url="http://172.19.0.1:11434",
        #     # model="qwen3:0.6b",
        #     base_url="http://100.107.76.9:11434",
        #     model="qwen3:8b",
        #     temperature=0.1,
        #     num_ctx=4096,
        #     reasoning=False,
        # )

        self.client = Client(host="http://100.107.76.9:11434")

    async def _generate_chunk_description(
        self,
        current_chunk_text: str,
        previous_summaries: list[str],
        filename: str,
    ) -> str:
        """
        Генерує стислий опис (контекст) для поточного чанку, базуючись на історії.
        """
        if previous_summaries:
            context_history = "\n".join([f"- {s}" for s in previous_summaries])
        else:
            context_history = "No previous context (Start of document)."

        prompt = f"""Create a concise contextual header for RAG retrieval indexing.

FORMAT: [Type] Entity Period: Specific Topics

RULES:
1. Type: [Announcement], [Table], [CEO Statement], [Chart], etc.

2. Entity & Period: 
   - Write naturally without extra brackets: "Ulta Beauty Q4 FY2022"
   - NOT: "[Ulta Beauty] [Q4 FY2022]"

3. Specific Topics (2-4 categories):
   - Be SPECIFIC: "Sales Growth & Margins" NOT "Revenue Metrics"
   - Distinguish from adjacent chunks - avoid repeating same categories
   - For tables with years: add "(3-year)" or "(multi-period)"
   
4. Length: Maximum 10-12 words total

GOOD EXAMPLES:
    - [Table] Ulta Beauty FY2022: Sales, Comparable Growth & Gross Margins (3-year)
    - [CEO Letter] Company X Q3 2023: Strategic Milestones & Market Outlook
    - [Announcement] Acme Corp Q1 2024: Revenue Beat & Guidance Raise

BAD EXAMPLES:
    - [Table] [Ulta Beauty] [Q4 FY2022]: Revenue Metrics, Profitability Indicators
    - [Announcement] Financial results and performance metrics

CONTEXT:
Previous summaries: {context_history}

CHUNK:
{current_chunk_text}

Generate header:"""

        try:
            response = self.client.chat(
                model="qwen3:8b-q4_K_M",
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating description via Ollama: {e}")
            return f"Fragment from {filename}"
