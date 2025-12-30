# app/services/file_extraction.py
"""
Service for extracting text from various file formats (PDF, DOCX, TXT).
"""
from typing import Optional
import io


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Install it with: pip install PyPDF2"
        )
    
    pdf_file = io.BytesIO(file_content)
    reader = PdfReader(pdf_file)
    
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        except Exception as e:
            print(f"[WARN] Failed to extract text from page {page_num + 1}: {e}")
            continue
    
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_content: Raw bytes of the DOCX file
        
    Returns:
        Extracted text as a string
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        )
    
    docx_file = io.BytesIO(file_content)
    doc = Document(docx_file)
    
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    
    return "\n\n".join(text_parts)


def extract_text_from_txt(file_content: bytes) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        file_content: Raw bytes of the text file
        
    Returns:
        Decoded text as a string
    """
    # Try common encodings
    for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
        try:
            return file_content.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    
    # Fallback: decode with errors ignored
    return file_content.decode('utf-8', errors='ignore')


def extract_text_from_file(
    file_content: bytes,
    filename: str,
    content_type: Optional[str] = None
) -> str:
    """
    Extract text from a file based on its extension or content type.
    
    Args:
        file_content: Raw bytes of the file
        filename: Name of the file (used to determine type)
        content_type: MIME type of the file (optional)
        
    Returns:
        Extracted text as a string
        
    Raises:
        ValueError: If file type is not supported
    """
    filename_lower = filename.lower()
    
    # Determine file type
    if filename_lower.endswith('.pdf') or content_type == 'application/pdf':
        return extract_text_from_pdf(file_content)
    
    elif filename_lower.endswith('.docx') or content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(file_content)
    
    elif filename_lower.endswith('.txt') or content_type == 'text/plain':
        return extract_text_from_txt(file_content)
    
    else:
        # Try to extract as text anyway
        try:
            return extract_text_from_txt(file_content)
        except:
            raise ValueError(
                f"Unsupported file type: {filename}. "
                f"Supported types: PDF (.pdf), DOCX (.docx), TXT (.txt)"
            )

