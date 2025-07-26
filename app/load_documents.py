import os
from pathlib import Path
from typing import List, Dict

# PDF loader via langchain-docling
from langchain_docling  import DoclingLoader
from docling.chunking import HybridChunker
# Unstructured for DOCX
from unstructured.partition.docx import partition_docx
# Pandas for Excel
import pandas as pd

DATA_DIR = Path("data")


def normalize_text(text: str) -> str:
    # Remove extra whitespace, fix encoding issues, etc.
    return ' '.join(text.split())


def load_pdf(file_path: Path) -> List[Dict]:
    loader = DoclingLoader(str(file_path))
    docs = loader.load()
    results = []
    chunker = HybridChunker()
    for doc in docs:
        # docling-docling returns a DoclingDocument in .dl_doc
        dl_doc = getattr(doc, 'dl_doc', None)
        if dl_doc is not None:
            for chunk in chunker.chunk(dl_doc):
                results.append({
                    "text": normalize_text(chunk.text),
                    "metadata": {
                        **chunk.metadata,
                        "source": str(file_path),
                        "document_type": "pdf",
                    }
                })
        else:
            # fallback: old style
            results.append({
                "text": normalize_text(doc.page_content),
                "metadata": {
                    **doc.metadata,
                    "source": str(file_path),
                    "document_type": "pdf",
                }
            })
    return results


def load_docx(file_path: Path) -> List[Dict]:
    # Try to use DoclingLoader + HybridChunker for DOCX
    try:
        loader = DoclingLoader(str(file_path))
        docs = loader.load()
        results = []
        chunker = HybridChunker()
        for doc in docs:
            dl_doc = getattr(doc, 'dl_doc', None)
            if dl_doc is not None:
                for chunk in chunker.chunk(dl_doc):
                    results.append({
                        "text": normalize_text(chunk.text),
                        "metadata": {
                            **chunk.metadata,
                            "source": str(file_path),
                            "document_type": "docx",
                        }
                    })
            else:
                results.append({
                    "text": normalize_text(doc.page_content),
                    "metadata": {
                        **doc.metadata,
                        "source": str(file_path),
                        "document_type": "docx",
                    }
                })
        return results
    except Exception:
        # Fallback to unstructured.partition.docx
        elements = partition_docx(filename=str(file_path))
        results = []
        for el in elements:
            if el.text.strip():
                results.append({
                    "text": normalize_text(el.text),
                    "metadata": {
                        "source": str(file_path),
                        "document_type": "docx",
                    }
                })
        return results


def load_excel(file_path: Path) -> List[Dict]:
    # Try DoclingLoader + HybridChunker for Excel (if supported in your docling-docling version)
    try:
        loader = DoclingLoader(str(file_path))
        docs = loader.load()
        results = []
        chunker = HybridChunker()
        for doc in docs:
            dl_doc = getattr(doc, 'dl_doc', None)
            if dl_doc is not None:
                for chunk in chunker.chunk(dl_doc):
                    results.append({
                        "text": normalize_text(chunk.text),
                        "metadata": {
                            **chunk.metadata,
                            "source": str(file_path),
                            "document_type": "excel",
                        }
                    })
            else:
                results.append({
                    "text": normalize_text(doc.page_content),
                    "metadata": {
                        **doc.metadata,
                        "source": str(file_path),
                        "document_type": "excel",
                    }
                })
        return results
    except Exception:
        # Fallback to pandas for Excel
        dfs = pd.read_excel(file_path, sheet_name=None)
        results = []
        for sheet, df in dfs.items():
            for idx, row in df.iterrows():
                row_text = ' | '.join(str(cell) for cell in row.values)
                results.append({
                    "text": normalize_text(row_text),
                    "metadata": {
                        "source": str(file_path),
                        "document_type": "excel",
                        "sheet": sheet,
                        "row": idx,
                    }
                })
        return results


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