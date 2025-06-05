import os
import chromadb
import pdfplumber
from typing import List, Tuple, Dict, Generator
import gc
import asyncio
import uuid
from langchain_text_splitters import CharacterTextSplitter
import pytesseract
import docx
import pandas as pd 
from sqlite_db.query import save_file_metadata_to_db


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
    file_id: str,
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
                "original_file_path": file_path,
                "file_id": file_id
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
    file_id: str,
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
    texts, metadatas = chunk_pages(file_path, pages_data, file_id ,chunk_size, chunk_overlap)
    
    print(f"Finished RAG preparation for {os.path.basename(file_path)}. Extracted {len(texts)} chunks.")
    return texts, metadatas






















#========================================================




def extract_pdf_pages_in_batches(
    pdf_path: str, 
    batch_size: int = 10
) -> Generator[List[Tuple[int, str]], None, None]:
    """
    Generator that yields batches of PDF pages as (page_number, text) tuples.
    This prevents loading all pages into memory at once.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            current_batch = []
            
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                # Fallback to OCR if minimal text found
                if not text or len(text.strip()) < 40:
                    print(f"Page {i} of {os.path.basename(pdf_path)}: Falling back to OCR.")
                    try:
                        pil_img = page.to_image(resolution=300).original
                        text_ocr = pytesseract.image_to_string(pil_img)
                        text = text_ocr if text_ocr and len(text_ocr.strip()) > len(text.strip() if text else "") else text
                    except Exception as ocr_err:
                        print(f"OCR failed for page {i}: {ocr_err}")
                
                current_batch.append((i, text.strip() if text else ""))
                
                # Yield batch when it reaches the specified size
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                    # Force garbage collection after each batch
                    gc.collect()
            
            # Yield remaining pages if any
            if current_batch:
                yield current_batch
                
    except Exception as e:
        print(f"Error processing PDF {os.path.basename(pdf_path)}: {e}")
        # Yield error as a single batch
        yield [(1, f"Error processing PDF: {e}")]


def chunk_batch_pages(
    batch_pages: List[Tuple[int, str]],
    file_path: str,
    file_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[Dict]]:
    """
    Chunks a batch of pages and returns texts and metadata.
    """
    texts: List[str] = []
    metadatas: List[Dict] = []
    file_name = os.path.basename(file_path)
    
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    for page_num, page_text in batch_pages:
        if not page_text or not page_text.strip():
            continue
            
        page_chunks = splitter.split_text(page_text)
        
        for chunk_idx, chunk_content in enumerate(page_chunks, start=1):
            if not chunk_content.strip():
                continue
                
            texts.append(chunk_content)
            
            metadata = {
                "source": file_name,
                "chunk_on_page_id": chunk_idx,
                "original_file_path": file_path,
                "file_id": file_id,
                "page_number": page_num
            }
            
            metadatas.append(metadata)
    
    return texts, metadatas


async def process_pdf_in_batches(
    file_path: str,
    file_id: str,
    collection,  # ChromaDB collection
    batch_size: int = 10,
    embedding_batch_size: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict:
    """
    Process PDF in batches to manage memory usage.
    
    Args:
        file_path: Path to the PDF file
        file_id: Unique identifier for the file
        collection: ChromaDB collection object
        batch_size: Number of pages to process at once
        embedding_batch_size: Number of chunks to embed at once
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        Dictionary with processing results
    """
    total_chunks = 0
    total_pages_processed = 0
    
    try:
        # BATCH PROCESSING LOOP - This is where each batch gets processed sequentially
        print(f"Starting batch processing for {os.path.basename(file_path)}")
        
        for batch_number, page_batch in enumerate(extract_pdf_pages_in_batches(file_path, batch_size), 1):
            if not page_batch:
                continue
                
            print(f"🔄 PROCESSING BATCH #{batch_number}: {len(page_batch)} pages (pages {page_batch[0][0]}-{page_batch[-1][0]})")
            
            # STEP 1: Chunk the current batch of pages
            print(f"   └── Step 1: Chunking pages {page_batch[0][0]}-{page_batch[-1][0]}")
            batch_texts, batch_metadatas = await asyncio.to_thread(
                chunk_batch_pages, 
                page_batch, 
                file_path, 
                file_id, 
                chunk_size, 
                chunk_overlap
            )
            
            total_pages_processed += len(page_batch)
            print(f"   └── Created {len(batch_texts)} chunks from this batch")
            
            if not batch_texts:
                print(f"   └── No chunks created from batch #{batch_number}, skipping embedding")
                continue
            
            # STEP 2: Embed chunks in smaller batches to manage memory
            print(f"   └── Step 2: Embedding {len(batch_texts)} chunks in sub-batches of {embedding_batch_size}")
            await embed_chunks_in_batches(
                batch_texts, 
                batch_metadatas, 
                collection, 
                embedding_batch_size
            )
            
            total_chunks += len(batch_texts)
            
            # STEP 3: Clear batch data and force garbage collection
            print(f"   └── Step 3: Cleaning up memory for batch #{batch_number}")
            del batch_texts, batch_metadatas, page_batch
            gc.collect()
            
            print(f"✅ BATCH #{batch_number} COMPLETE: Total processed = {total_pages_processed} pages, {total_chunks} chunks")
            print(f"{'='*60}")
        
        print(f"🎉 ALL BATCHES COMPLETE: {total_pages_processed} pages, {total_chunks} chunks embedded")
        return {
            "status": "successfully_embedded",
            "chunks_embedded": total_chunks,
            "pages_processed": total_pages_processed,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error processing PDF {os.path.basename(file_path)}: {str(e)}"
        print(error_msg)
        return {
            "status": "error_embedding",
            "chunks_embedded": total_chunks,
            "pages_processed": total_pages_processed,
            "error": error_msg
        }


async def embed_chunks_in_batches(
    texts: List[str],
    metadatas: List[Dict],
    collection,
    batch_size: int = 100
) -> None:
    """
    Embed chunks in smaller batches to manage memory.
    This creates SUB-BATCHES within each page batch for embedding.
    """
    total_embedding_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"      └── Will create {total_embedding_batches} embedding sub-batches of max {batch_size} chunks each")
    
    for sub_batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch_texts]
        
        print(f"      └── Embedding sub-batch {sub_batch_num}/{total_embedding_batches}: {len(batch_texts)} chunks (chunks {i+1}-{i+len(batch_texts)})")
        
        # Embed the current sub-batch
        await asyncio.to_thread(
            collection.add,
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        
        print(f"      └── ✅ Sub-batch {sub_batch_num} embedded successfully")
        
        # Clear sub-batch data and force garbage collection
        del batch_texts, batch_metadatas, batch_ids
        gc.collect()
    



async def process_and_embed_pdf_batched(
    file_path: str, 
    original_filename: str,
    collection,
    page_batch_size: int = 10,
    embedding_batch_size: int = 100
) -> Dict:
    """
    Updated version of process_and_embed_file specifically for PDFs with batch processing.
    """
    print(f"Processing and embedding PDF in batches: {original_filename}")
    
    file_result = {
        "filename": original_filename,
        "path": file_path,
        "status": "processing",
        "chunks_embedded": 0,
        "pages_processed": 0,
        "error": None
    }
    
    file_id = str(uuid.uuid4())
    
    try:
        # Check if file is PDF
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("This function is specifically for PDF files")
        
        # Process PDF in batches
        result = await process_pdf_in_batches(
            file_path=file_path,
            file_id=file_id,
            collection=collection,
            batch_size=page_batch_size,
            embedding_batch_size=embedding_batch_size
        )
        
        file_result.update(result)
        
        if result["status"] == "successfully_embedded":
            print(f"Successfully embedded {result['chunks_embedded']} chunks from {original_filename}")
            
            # Save file metadata
            await save_file_metadata_to_db(
                file_id=file_id,
                filename=original_filename,
                path=file_path,
                content_type="application/pdf",
                chunk_count=result["chunks_embedded"]
            )
        
    except Exception as e:
        error_message = f"Error during batch embedding process for {original_filename}: {str(e)}"
        print(error_message)
        
        file_result["status"] = "error_embedding"
        file_result["error"] = error_message
        
        # Clean up file if needed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up failed file: {file_path}")
            except Exception as rm_err:
                print(f"Error cleaning up file {file_path}: {rm_err}")
    
    return file_result


