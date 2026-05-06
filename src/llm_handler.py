# src/llm_handler.py
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OLLAMA_BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL,
    GEMINI_API_KEY, GROQ_API_KEY
)


def build_prompt(question: str, context_chunks: list, chat_history: list = []) -> str:
    """
    Builds prompt with:
    - Chat history so model remembers previous messages
    - Retrieved context from PDF
    - User's question in whatever format they want
    """
    context = "\n\n---\n\n".join(context_chunks)

    # Build conversation history — last 6 messages only
    history_text = ""
    if chat_history:
        history_text = "PREVIOUS CONVERSATION:\n"
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Only include text content, skip context metadata
            content = msg["content"]
            if content and len(content) > 0:
                history_text += f"{role}: {content}\n"
        history_text += "\n"

    prompt = f"""You are an intelligent document assistant helping a user understand a document.

RULES:
- Use the provided document context as your PRIMARY source of information
- Answer in whatever format the user requests (100 words, bullet points, table, summary, simple English, etc.)
- If the user asks for a specific word count or format — follow it STRICTLY
- Synthesize and explain information from the context in your own words
- Use the previous conversation to understand follow-up questions and pronouns like "he", "it", "they"
- If the topic is completely absent from the context, say "This topic is not covered in the document"
- Never say you cannot find something if the context clearly contains related information
- Be conversational, helpful and clear

{history_text}CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""

    return prompt


def get_answer(question: str, context_chunks: list,
               model_name: str = DEFAULT_MODEL,
               chat_history: list = []) -> str:
    """
    Routes to correct provider. Passes chat history for memory.
    """
    if not context_chunks:
        return "No relevant content found in the document to answer your question."

    model_config = AVAILABLE_MODELS.get(model_name)
    if not model_config:
        return f"Model '{model_name}' not found in configuration."

    prompt = build_prompt(question, context_chunks, chat_history)
    model_type = model_config["type"]

    if model_type == "ollama":
        return _call_ollama(prompt, model_config["model"])
    elif model_type == "gemini":
        return _call_gemini(prompt, model_config["model"])
    elif model_type == "groq":
        return _call_groq(prompt, model_config["model"])
    else:
        return f"Unknown model type: {model_type}"


def _call_ollama(prompt: str, model: str) -> str:
    """Calls local Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        if response.status_code != 200:
            return f"❌ Ollama error: {response.text}"
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "❌ Model timed out. Try a smaller/faster model."
    except Exception as e:
        return f"❌ Ollama error: {str(e)}"


def _call_gemini(prompt: str, model: str) -> str:
    """Calls Google Gemini API."""
    try:
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            return "❌ GEMINI_API_KEY not found in .env file."
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        error = str(e)
        if "quota" in error.lower() or "429" in error:
            return "❌ Gemini quota exceeded. Switch to a local model or try later."
        if "API_KEY" in error.upper():
            return "❌ Invalid Gemini API key. Check your .env file."
        return f"❌ Gemini error: {error}"


def _call_groq(prompt: str, model: str) -> str:
    """Calls Groq Cloud API."""
    try:
        if not GROQ_API_KEY:
            return "❌ GROQ_API_KEY not found in .env file."
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024
            },
            timeout=30
        )
        if response.status_code == 401:
            return "❌ Invalid Groq API key. Check your .env file."
        if response.status_code == 429:
            return "❌ Groq rate limit hit. Wait a moment and try again."
        if response.status_code != 200:
            return f"❌ Groq error: {response.text}"
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ Groq request timed out. Try again."
    except Exception as e:
        return f"❌ Groq error: {str(e)}"


def get_available_models() -> dict:
    return AVAILABLE_MODELS


def check_model_available(model_name: str) -> bool:
    model_config = AVAILABLE_MODELS.get(model_name, {})
    model_type = model_config.get("type")
    if model_type == "ollama":
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if response.status_code != 200:
                return False
            pulled = [m["name"] for m in response.json().get("models", [])]
            model_id = model_config["model"]
            return any(model_id in m for m in pulled)
        except:
            return False
    elif model_type == "gemini":
        return bool(GEMINI_API_KEY)
    elif model_type == "groq":
        return bool(GROQ_API_KEY)
    return False


def check_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False