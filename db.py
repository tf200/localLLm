import chromadb
import pdfplumber
from typing import List, Tuple, Dict, Optional, Union
from langchain_text_splitters import CharacterTextSplitter
import os
import uuid
import docx
# For Excel files
import pandas as pd
import openpyxl
# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("Micla")

def get_file_extension(file_path: str) -> str:
    """
    Extracts the lowercase file extension from a file path.
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
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

def extract_text_from_docx(docx_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from a Word document (.docx).
    Simulates "pages" by splitting on form feeds or by paragraph count.
    Returns a list of tuples: (page_number, page_text).
    """
    doc = docx.Document(docx_path)
    
    # Get all paragraphs as text
    paragraphs = [p.text for p in doc.paragraphs]
    
    # First try to find form feed characters which might indicate page breaks
    full_text = '\n'.join(paragraphs)
    if '\f' in full_text:
        # Split by form feed character which represents page breaks
        page_texts = full_text.split('\f')
        return [(i, text) for i, text in enumerate(page_texts, start=1)]
    
    # If no form feeds, create artificial "pages" of roughly 3000 characters
    # (~15-20 paragraphs per page as a rough estimate)
    artificial_pages = []
    current_page = []
    current_length = 0
    page_number = 1
    
    for paragraph in paragraphs:
        current_page.append(paragraph)
        current_length += len(paragraph)
        
        # Create a new page after approximately 3000 characters
        if current_length > 3000:
            artificial_pages.append((page_number, '\n'.join(current_page)))
            current_page = []
            current_length = 0
            page_number += 1
    
    # Add the last page if there's anything remaining
    if current_page:
        artificial_pages.append((page_number, '\n'.join(current_page)))
    
    return artificial_pages

def extract_text_from_doc(doc_path: str) -> List[Tuple[int, str]]:
    try:
        import docx2txt
        text = docx2txt.process(doc_path)
        
        # Split by form-feed (page breaks) or artificial pages
        if "\f" in text:
            return [(i, page) for i, page in enumerate(text.split("\f"), 1)]
        else:
            return [(i, text[i:i+3000]) for i in range(0, len(text), 3000)]
    
    except ImportError:
        return [(1, "ERROR: Install 'python-docx2txt' for .doc support.")]

def extract_text_from_excel(excel_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from Excel files (.xls, .xlsx).
    Each sheet is treated as a "page".
    Returns a list of tuples: (sheet_number, sheet_text_representation).
    """
    # Load the Excel file
    excel_file = pd.ExcelFile(excel_path)
    sheets = []
    
    for i, sheet_name in enumerate(excel_file.sheet_names, start=1):
        # Read the sheet into a DataFrame
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Convert DataFrame to text representation
        # This adds column headers and formats rows
        rows = []
        
        # Add column headers
        headers = " | ".join(str(col) for col in df.columns)
        rows.append(headers)
        rows.append("-" * len(headers))
        
        # Add data rows
        for _, row in df.iterrows():
            row_text = " | ".join(str(cell) for cell in row)
            rows.append(row_text)
        
        sheet_text = f"Sheet: {sheet_name}\n" + "\n".join(rows)
        sheets.append((i, sheet_text))
    
    return sheets

def extract_text_by_page(file_path: str) -> List[Tuple[int, str]]:
    """
    Universal extractor that determines file type and calls appropriate extractor.
    Returns a list of tuples: (page_number/sheet_number, text).
    """
    extension = get_file_extension(file_path)
    
    # Dispatch based on file extension
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.doc':
        return extract_text_from_doc(file_path)
    elif extension in ['.xlsx', '.xls']:
        return extract_text_from_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def chunk_pages(
    file_path: str,
    pages: List[Tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Splits page texts into overlapping chunks.
    Works for any document type.
    Returns two lists: texts and corresponding metadata dicts.
    """
    texts: List[str] = []
    metadatas: List[Dict] = []
    file_name = os.path.basename(file_path)
    extension = get_file_extension(file_path)

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
            
            # Customize metadata based on file type
            metadata = {
                "source": file_name,
                "chunk_id": idx
            }
            
            # Add file-type specific metadata
            if extension in ['.pdf', '.doc', '.docx']:
                metadata["page"] = page_num
            elif extension in ['.xlsx', '.xls']:
                metadata["sheet"] = page_num
            
            metadatas.append(metadata)
            
    return texts, metadatas

def prepare_document_for_rag(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Full pipeline: extract content, chunk it, and return texts + metadata.
    Works with PDF, Word, and Excel files.
    """
    pages = extract_text_by_page(file_path)
    texts, metadatas = chunk_pages(file_path, pages, chunk_size, chunk_overlap)
    return texts, metadatas

# def process_all_pdfs(directory: str = "files", chunk_size: int = 1500, chunk_overlap: int = 300):
#     """
#     Process all PDF files in the given directory and add them to the ChromaDB collection.
#     """
#     # Get all PDF files in the directory
#     pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         print(f"No PDF files found in directory: {directory}")
#         return
    
#     total_chunks_processed = 0
    
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(directory, pdf_file)
#         print(f"\nProcessing file: {pdf_file}")
        
#         try:
#             # Prepare the PDF for RAG
#             texts, metadatas = prepare_pdf_for_rag(
#                 pdf_path,
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap
#             )
            
#             total_chunks = len(texts)
#             total_chunks_processed += total_chunks
            
#             print(f"Prepared {total_chunks} chunks from {pdf_file}")
#             if total_chunks > 0:
#                 print("First chunk preview:")
#                 print(texts[0][:200], "...")
#                 print("Metadata:", metadatas[0])
                
#                 # Generate unique IDs and add to collection
#                 ids = [str(uuid.uuid4()) for _ in texts]
#                 collection.add(documents=texts, metadatas=metadatas, ids=ids)
#                 print(f"Successfully added {total_chunks} chunks to the collection")
#             else:
#                 print(f"No text chunks were created for {pdf_file}")
                
#         except Exception as e:
#             print(f"Error processing file {pdf_file}: {str(e)}")
    
#     print(f"\nProcessing complete. Total chunks processed across all files: {total_chunks_processed}")

# # Process all PDFs in the 'files' directory
# process_all_pdfs()