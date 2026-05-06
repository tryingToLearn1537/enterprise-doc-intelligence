# EDIS — Enterprise Document Intelligence System 🧠

**EDIS** is a high-performance Retrieval-Augmented Generation (RAG) platform designed to transform static PDF documents into interactive, searchable intelligence. It enables users to upload complex documents and engage in context-aware conversations powered by local and cloud-based AI models.

## 🚀 Key Features
*   **Multi-Model Support**: Seamlessly switch between local models via Ollama and cloud providers like Google Gemini and Groq[cite: 5].
*   **Semantic Vector Search**: High-accuracy retrieval using `sentence-transformers` and `ChromaDB`[cite: 4, 8].
*   **Smart PDF Processing**: Automated text extraction and recursive character chunking with overlap preservation[cite: 6].
*   **Global Chat Timeline**: Maintains a unified conversation history across different models in a single view.
*   **Transparency**: Includes expandable "Source Chunks" to show the exact document context used for each answer[cite: 3, 5].

## 🛠️ Tech Stack
*   **Frontend**: Streamlit[cite: 3]
*   **Vector Database**: ChromaDB[cite: 8]
*   **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)[cite: 4]
*   **PDF Engine**: PyMuPDF (`fitz`)[cite: 6]
*   **Orchestration**: LangChain[cite: 3, 6]
*   **LLM Providers**: Ollama, Google Gemini, Groq[cite: 5]

## 📋 Project Structure
*   `app.py`: The main Streamlit interface and UI logic[cite: 3].
*   `src/rag_pipeline.py`: Orchestrates the ingestion and retrieval flow[cite: 7].
*   `src/pdf_processor.py`: Handles PDF text extraction and chunking[cite: 6].
*   `src/vector_store.py`: Manages ChromaDB collections and similarity search[cite: 8].
*   `src/llm_handler.py`: Routes prompts to various LLM providers[cite: 5].
*   `src/embeddings.py`: Generates vector representations of text[cite: 4].
*   `utils/helpers.py`: Session state management and UI utilities[cite: 2].

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) (Required for local model support)

### 2. Environment Setup
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

3. Installation
# Install required dependencies
pip install -r requirements.txt

4. Running the App
streamlit run app.py

How to Use
1.Select a Model: Use the sidebar to pick your preferred AI engine[cite: 3].
2.Upload PDF: Drop a document into the sidebar; the system will index it into the vector store.
3.Chat: Ask questions about your document. EDIS retrieves relevant sections and generates an answer[cite: 3, 5].
4.Verify: Expand "Source Chunks" below the AI response to see the evidence used[cite: 3].
