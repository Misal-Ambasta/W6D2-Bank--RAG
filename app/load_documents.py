import os
from pathlib import Path
from typing import List, Dict

# LangChain text splitter for recursive chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Lightweight PDF processing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Fallback: langchain-docling (memory intensive)
try:
    from langchain_docling import DoclingLoader
    from docling.chunking import HybridChunker
except ImportError:
    DoclingLoader = None
    HybridChunker = None

# Unstructured for DOCX
from unstructured.partition.docx import partition_docx
# Pandas for Excel
import pandas as pd

DATA_DIR = Path("data")


def normalize_text(text: str) -> str:
    """Remove extra whitespace, fix encoding issues, etc."""
    return ' '.join(text.split())

# Initialize the recursive text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def split_text_into_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    if not text or not text.strip():
        return []
    
    # Use LangChain's recursive splitter for better chunk quality
    chunks = text_splitter.split_text(text)
    
    # Filter out empty chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def load_pdf(file_path: Path) -> List[Dict]:
    """Load PDF using lightweight PyPDF2 approach to avoid memory issues."""
    results = []
    
    # Use PyPDF2 for memory-efficient PDF reading (NO FALLBACK to avoid memory issues)
    if not PdfReader:
        print(f"PyPDF2 not available. Cannot process PDF: {file_path}")
        return results
    
    try:
        print(f"Processing PDF with PyPDF2: {file_path}")
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        print(f"PDF has {total_pages} pages")
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():  # Only add non-empty pages
                    print(f"Extracted text from page {page_num + 1} ({len(text)} characters)")
                    # Split long pages into smaller chunks (max 1000 chars)
                    chunks = split_text_into_chunks(text, max_chunk_size=1000)
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():  # Only add non-empty chunks
                            results.append({
                                "text": normalize_text(chunk),
                                "metadata": {
                                    "source": str(file_path),
                                    "document_type": "pdf",
                                    "page": page_num + 1,
                                    "chunk": chunk_idx + 1,
                                    "total_pages": total_pages,
                                }
                            })
                else:
                    print(f"Page {page_num + 1} is empty or has no extractable text")
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                continue
        
        print(f"Successfully processed PDF: {len(results)} chunks extracted")
        return results
        
    except Exception as e:
        print(f"PyPDF2 failed for {file_path}: {e}")
        print(f"Could not process PDF: {file_path}")
        return []


def load_docx(file_path: Path) -> List[Dict]:
    """Load DOCX using lightweight unstructured approach to avoid memory issues."""
    print(f"Processing DOCX with unstructured: {file_path}")
    try:
        # Use unstructured.partition.docx (lightweight approach)
        elements = partition_docx(filename=str(file_path))
        results = []
        for idx, el in enumerate(elements):
            if el.text and el.text.strip():
                # Split long elements into smaller chunks
                chunks = split_text_into_chunks(el.text, max_chunk_size=1000)
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.strip():
                        results.append({
                            "text": normalize_text(chunk),
                            "metadata": {
                                "source": str(file_path),
                                "document_type": "docx",
                                "element": idx + 1,
                                "chunk": chunk_idx + 1,
                            }
                        })
        print(f"Successfully processed DOCX: {len(results)} chunks extracted")
        return results
    except Exception as e:
        print(f"Failed to process DOCX {file_path}: {e}")
        return []


def load_excel(file_path: Path) -> List[Dict]:
    """Load Excel using lightweight pandas approach to avoid memory issues."""
    print(f"Processing Excel with pandas: {file_path}")
    try:
        # Use pandas for Excel (lightweight approach)
        dfs = pd.read_excel(file_path, sheet_name=None)
        results = []
        for sheet, df in dfs.items():
            print(f"Processing sheet '{sheet}' with {len(df)} rows")
            for idx, row in df.iterrows():
                row_text = ' | '.join(str(cell) for cell in row.values if pd.notna(cell))
                if row_text.strip():
                    # Split long rows into smaller chunks if needed
                    chunks = split_text_into_chunks(row_text, max_chunk_size=1000)
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():
                            results.append({
                                "text": normalize_text(chunk),
                                "metadata": {
                                    "source": str(file_path),
                                    "document_type": "excel",
                                    "sheet": sheet,
                                    "row": idx + 1,
                                    "chunk": chunk_idx + 1,
                                }
                            })
        print(f"Successfully processed Excel: {len(results)} chunks extracted")
        return results
    except Exception as e:
        print(f"Failed to process Excel {file_path}: {e}")
        return []


def main():
    all_chunks = []
    for file in DATA_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            print(f"Loading PDF: {file}")
            all_chunks.extend(load_pdf(file))
        elif file.suffix.lower() == ".docx":
            print(f"Loading DOCX: {file}")
            all_chunks.extend(load_docx(file))
        elif file.suffix.lower() in [".xls", ".xlsx"]:
            print(f"Loading Excel: {file}")
            all_chunks.extend(load_excel(file))
    print(f"\nLoaded {len(all_chunks)} chunks.")
    # Print a summary of the first 3 chunks
    for chunk in all_chunks[:3]:
        print("\n--- Chunk ---")
        print(f"Text: {chunk['text'][:200]}...")
        print(f"Metadata: {chunk['metadata']}")

if __name__ == "__main__":
    main() 