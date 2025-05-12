import chromadb
import pdfplumber
from typing import List, Tuple, Dict
from langchain_text_splitters import CharacterTextSplitter
import os



client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("Micla")






def extract_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from each page of the PDF.
    Returns a list of tuples: (page_number, page_text).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages





USE_LANGCHAIN = True  #
def chunk_pages(
    pdf_path: str,
    pages: List[Tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Splits page texts into overlapping chunks.
    Returns two lists: texts and corresponding metadata dicts.
    Metadata includes: source filename, page number, chunk index.
    """
    texts: List[str] = []
    metadatas: List[Dict] = []

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for page_num, text in pages:
        if not text.strip():
            continue

        # Use langchain splitter
        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks, start=1):
            texts.append(chunk)
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk_id": idx
            })
    return texts, metadatas



def prepare_pdf_for_rag(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Full pipeline: extract pages, chunk them, and return texts + metadata.
    """
    pages = extract_text_by_page(pdf_path)
    texts, metadatas = chunk_pages(pdf_path, pages, chunk_size, chunk_overlap)
    return texts, metadatas



pdf_file = "files/auto_elec_cdx.pdf"


texts, metadatas = prepare_pdf_for_rag(
    pdf_file,
    chunk_size=1500,
    chunk_overlap=300
)
total_chunks = len(texts)
print(f"Prepared {total_chunks} chunks from {os.path.basename(pdf_file)}")
if total_chunks > 0:
    print("First chunk preview:")
    print(texts[0][:200], "...")
    print("Metadata:", metadatas[0])



import uuid

ids = [str(uuid.uuid4()) for _ in texts]
collection.add(documents=texts, metadatas=metadatas, ids=ids)