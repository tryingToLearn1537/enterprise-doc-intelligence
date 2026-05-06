# src/pdf_processor.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import clean_text  # ← NEW IMPORT


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts all the text from every page.
    Returns one big cleaned string with all the text.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n"
        full_text += text

    doc.close()
    return clean_text(full_text)  # ← CHANGED: clean before returning


def split_text_into_chunks(text: str) -> list:
    """
    Takes the big text string and cuts it into smaller overlapping chunks.
    Returns a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def process_pdf(pdf_path: str) -> list:
    """
    Master function: takes a PDF path, returns a list of text chunks.
    This is what other files will call.
    """
    print(f"📄 Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text found in PDF. It might be a scanned image PDF.")

    print(f"✅ Extracted {len(text)} characters from PDF")

    chunks = split_text_into_chunks(text)
    print(f"✅ Split into {len(chunks)} chunks")

    return chunks


# --- TEST ---
if __name__ == "__main__":
    import sys

    test_pdf_path = "test_sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test PDF for the Enterprise Document Intelligence System. " * 20)
    doc.save(test_pdf_path)
    doc.close()

    chunks = process_pdf(test_pdf_path)
    print(f"\n--- First Chunk Preview ---")
    print(chunks[0])
    print(f"\n--- Total Chunks: {len(chunks)} ---")

    os.remove(test_pdf_path)
    print("\n🎉 pdf_processor.py is working correctly!")