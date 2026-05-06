# src/rag_pipeline.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_processor import process_pdf
from src.vector_store import store_chunks, search_similar_chunks, clear_collection, get_collection_count
from src.llm_handler import get_answer, check_ollama_running
from config import DEFAULT_MODEL


def ingest_pdf(pdf_path: str, doc_name: str) -> dict:
    """PDF → chunks → embeddings → ChromaDB"""
    try:
        chunks = process_pdf(pdf_path)
        clear_collection()
        store_chunks(chunks, doc_name)
        return {
            "success": True,
            "chunks": len(chunks),
            "message": f"✅ Successfully processed {doc_name} into {len(chunks)} chunks"
        }
    except Exception as e:
        return {
            "success": False,
            "chunks": 0,
            "message": f"❌ Error processing PDF: {str(e)}"
        }


def answer_question(question: str,
                    model_name: str = DEFAULT_MODEL,
                    chat_history: list = []) -> dict:
    """
    Question → search → retrieve → LLM → answer
    chat_history: list of previous messages for this model
    """
    try:
        if not check_ollama_running():
            # Only block if an ollama model is selected
            from config import AVAILABLE_MODELS
            model_type = AVAILABLE_MODELS.get(model_name, {}).get("type")
            if model_type == "ollama":
                return {
                    "success": False,
                    "answer": "❌ Ollama is not running. Please start Ollama.",
                    "context": []
                }

        if get_collection_count() == 0:
            return {
                "success": False,
                "answer": "⚠️ No document loaded. Please upload a PDF first.",
                "context": []
            }

        # Retrieve relevant chunks
        context_chunks = search_similar_chunks(question)

        # Get answer with memory of previous messages
        answer = get_answer(
            question,
            context_chunks,
            model_name,
            chat_history  # ← pass history here
        )

        return {
            "success": True,
            "answer": answer,
            "context": context_chunks
        }

    except Exception as e:
        return {
            "success": False,
            "answer": f"❌ Error: {str(e)}",
            "context": []
        }