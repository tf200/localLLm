import os
import chromadb
import pdfplumber
from typing import List, Tuple, Dict
from langchain_text_splitters import CharacterTextSplitter
import pytesseract
import docx
import pandas as pd

# --- ChromaDB Initialization ---
# Adjust the path if your 'db' directory is located elsewhere relative to this file.
# For the suggested structure, if main.py runs from the root, this path should be "db"
# If embedding_utils.py is in a subdirectory, adjust accordingly.
# For the provided structure, running main.py from root:
DB_PATH = "db" # Path relative to where the main script is run
COLLECTION_NAME = "Micla"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

print(f"ChromaDB client initialized. Path: {os.path.abspath(DB_PATH)}, Collection: {COLLECTION_NAME}")


# --- Helper Functions for Text Extraction and Processing ---

def get_file_extension(file_path: str) -> str:
    """
    Extracts the lowercase file extension from a file path.
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from each page of the PDF.
    Falls back to OCR if no text is extracted.
    Returns a list of tuples: (page_number, page_text).
    """
    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text or len(text.strip()) < 40: # Basic check for meaningful text
                    print(f"Page {i} of {os.path.basename(pdf_path)}: Text extraction minimal or failed, falling back to OCR.")
                    try:
                        pil_img = page.to_image(resolution=300).original
                        text_ocr = pytesseract.image_to_string(pil_img)
                        text = text_ocr if text_ocr and len(text_ocr.strip()) > len(text.strip() if text else "") else text
                    except Exception as ocr_err:
                        print(f"OCR failed for page {i} of {os.path.basename(pdf_path)}: {ocr_err}")
                        # Keep original extracted text if OCR fails
                pages_data.append((i, text.strip() if text else ""))
    except Exception as e:
        print(f"Error processing PDF {os.path.basename(pdf_path)}: {e}")
        # Return a single "page" with error info if the whole PDF fails to open
        return [(1, f"Error processing PDF: {e}")]
    
    if not pages_data:
        print(f"No text could be extracted from {os.path.basename(pdf_path)}.")
        return [(1, "No text extracted from PDF.")]
        
    # Debug print for a specific page if needed and exists
    # if len(pages_data) > 5:
    #     print(f"Debug (PDF): Page 6 content (first 50 chars) from {os.path.basename(pdf_path)}: {pages_data[5][1][:50]}")
    return pages_data

def extract_text_from_docx(docx_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from a Word document (.docx).
    Simulates "pages" by splitting on form feeds or by paragraph count.
    Returns a list of tuples: (page_number, page_text).
    """
    try:
        doc = docx.Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs]
        full_text = '\n'.join(paragraphs)

        if '\f' in full_text: # Form feed character for page breaks
            page_texts = full_text.split('\f')
            return [(i, text.strip()) for i, text in enumerate(page_texts, start=1)]
        
        # If no form feeds, create artificial "pages" of roughly 3000 characters
        artificial_pages = []
        current_page_text = []
        current_length = 0
        page_number = 1
        
        for paragraph in paragraphs:
            current_page_text.append(paragraph)
            current_length += len(paragraph) + 1 # +1 for newline
            
            if current_length > 3000 and current_page_text:
                artificial_pages.append((page_number, '\n'.join(current_page_text).strip()))
                current_page_text = []
                current_length = 0
                page_number += 1
        
        if current_page_text: # Add any remaining text
            artificial_pages.append((page_number, '\n'.join(current_page_text).strip()))
        
        return artificial_pages if artificial_pages else [(1, full_text.strip())]
    except Exception as e:
        print(f"Error processing DOCX {os.path.basename(docx_path)}: {e}")
        return [(1, f"Error processing DOCX: {e}")]


def extract_text_from_doc(doc_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from a Word document (.doc).
    Requires python-docx2txt.
    """
    try:
        import docx2txt # Import here to make dependency optional if not used
        text = docx2txt.process(doc_path)
        
        if "\f" in text: # Form feed character often used for page breaks
            page_texts = text.split("\f")
            return [(i, page.strip()) for i, page in enumerate(page_texts, start=1)]
        else:
            # Create artificial "pages" if no form feeds, approx 3000 chars
            # This is a very rough approximation
            page_size = 3000
            num_pages = (len(text) + page_size - 1) // page_size
            return [(i + 1, text[i*page_size:(i+1)*page_size].strip()) for i in range(num_pages)] if text else [(1,"")]
    
    except ImportError:
        error_msg = "ERROR: 'docx2txt' library not installed. Please install it for .doc file support (`pip install docx2txt`)."
        print(error_msg)
        return [(1, error_msg)]
    except Exception as e:
        error_msg = f"Error processing .doc file {os.path.basename(doc_path)}: {e}"
        print(error_msg)
        return [(1, error_msg)]

def extract_text_from_excel(excel_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from Excel files (.xls, .xlsx).
    Each sheet is treated as a "page".
    Returns a list of tuples: (sheet_number, sheet_text_representation).
    """
    try:
        excel_file = pd.ExcelFile(excel_path)
        sheets_data = []
        
        for i, sheet_name in enumerate(excel_file.sheet_names, start=1):
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None) # Read without auto-header
            
            # Convert DataFrame to a simple text representation
            # This aims to capture all cell content as strings
            sheet_texts = []
            for _, row in df.iterrows():
                row_texts = [str(cell) if pd.notna(cell) else "" for cell in row]
                sheet_texts.append(" | ".join(row_texts)) # Join cells with a separator
            
            sheet_content = "\n".join(sheet_texts)
            # Prepend sheet name for context
            full_sheet_text = f"Sheet: {sheet_name}\n{sheet_content}".strip()
            sheets_data.append((i, full_sheet_text))
        
        return sheets_data if sheets_data else [(1, "No data found in Excel file.")]
    except Exception as e:
        print(f"Error processing Excel file {os.path.basename(excel_path)}: {e}")
        return [(1, f"Error processing Excel: {e}")]


def extract_text_by_page(file_path: str) -> List[Tuple[int, str]]:
    """
    Universal extractor that determines file type and calls appropriate extractor.
    Returns a list of tuples: (page_number/sheet_number, text).
    """
    extension = get_file_extension(file_path)
    print(f"Extracting text from {os.path.basename(file_path)} (type: {extension})")
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.doc':
        return extract_text_from_doc(file_path)
    elif extension in ['.xlsx', '.xls']:
        return extract_text_from_excel(file_path)
    # Add .txt support
    elif extension == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Treat the whole TXT file as a single "page"
            return [(1, content.strip())]
        except Exception as e:
            print(f"Error reading text file {os.path.basename(file_path)}: {e}")
            return [(1, f"Error reading text file: {e}")]
    else:
        unsupported_msg = f"Unsupported file format: {extension} for file {os.path.basename(file_path)}"
        print(unsupported_msg)
        # Return a "page" with the error message for graceful handling downstream
        return [(1, unsupported_msg)]


def chunk_pages(
    file_path: str,
    pages_data: List[Tuple[int, str]], # Renamed from 'pages' to 'pages_data' for clarity
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Splits page texts into overlapping chunks.
    Returns two lists: texts (chunks) and corresponding metadata dicts.
    """
    texts: List[str] = []
    metadatas: List[Dict] = []
    file_name = os.path.basename(file_path)
    extension = get_file_extension(file_path)

    # Using a more generic separator, but consider if specific ones are better for your data
    splitter = CharacterTextSplitter(
        separator="\n\n",  # Common paragraph separator
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # Keep False unless using regex separator
    )

    for page_num, page_text in pages_data:
        if not page_text or not page_text.strip(): # Skip empty or whitespace-only pages/sheets
            print(f"Skipping empty page/sheet {page_num} from {file_name}")
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_idx, chunk_content in enumerate(page_chunks, start=1):
            if not chunk_content.strip(): # Skip empty chunks that might result from splitting
                continue
            texts.append(chunk_content)
            
            metadata = {
                "source": file_name,
                "chunk_on_page_id": chunk_idx, # Distinguishes chunks from the same page/sheet
                 # Add original file path for potential future reference, if desired
                "original_file_path": file_path 
            }
            
            # Add file-type specific metadata key for page/sheet number
            if extension in ['.pdf', '.doc', '.docx', '.txt']:
                metadata["page_number"] = page_num
            elif extension in ['.xlsx', '.xls']:
                metadata["sheet_number"] = page_num # page_num is sheet_number for Excel
            
            metadatas.append(metadata)
            
    return texts, metadatas


def prepare_document_for_rag(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Full pipeline: extract content by page/sheet, chunk it, and return texts + metadata.
    """
    print(f"Starting RAG preparation for: {os.path.basename(file_path)}")
    
    # Step 1: Extract text content organized by page/sheet
    pages_data = extract_text_by_page(file_path)
    
    if not pages_data or (len(pages_data) == 1 and not pages_data[0][1].strip()):
        print(f"No meaningful text extracted from {os.path.basename(file_path)}. Skipping chunking.")
        return [], [] # Return empty lists if no text was extracted

    # Step 2: Chunk the extracted page/sheet texts
    texts, metadatas = chunk_pages(file_path, pages_data, chunk_size, chunk_overlap)
    
    print(f"Finished RAG preparation for {os.path.basename(file_path)}. Extracted {len(texts)} chunks.")
    return texts, metadatas