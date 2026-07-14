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
        
    Raises:
        ImportError: If required libraries are not installed
        ValueError: If PDF is encrypted and cannot be decrypted
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Install it with: pip install PyPDF2"
        )
    
    pdf_file = io.BytesIO(file_content)
    
    try:
        reader = PdfReader(pdf_file)
        
        # Check if PDF is encrypted
        if reader.is_encrypted:
            # Try to decrypt with empty password (common for some PDFs)
            try:
                reader.decrypt("")
                print("[INFO] PDF was encrypted but decrypted with empty password")
            except Exception as decrypt_error:
                # If decryption fails, provide helpful error message
                raise ValueError(
                    "This PDF file is password-protected. Please provide an unencrypted version or "
                    "the password-protected PDF may need special handling. "
                    f"Decryption error: {str(decrypt_error)}"
                )
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            except Exception as e:
                print(f"[WARN] Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            raise ValueError(
                "Could not extract any text from the PDF. "
                "The PDF may be image-based (scanned) or corrupted. "
                "Please try converting it to text first or use an OCR tool."
            )
        
        return "\n\n".join(text_parts)
    
    except ImportError as ie:
        # Check if it's the PyCryptodome error
        if "PyCryptodome" in str(ie) or "Crypto" in str(ie):
            raise ImportError(
                "This PDF requires encryption support. "
                "Please install PyCryptodome: pip install pycryptodome"
            )
        raise ie
    except Exception as e:
        # Catch any other PDF processing errors
        error_msg = str(e)
        if "PyCryptodome" in error_msg or "Crypto" in error_msg:
            raise ImportError(
                "This PDF requires encryption support. "
                "Please install PyCryptodome: pip install pycryptodome"
            )
        raise ValueError(f"Failed to process PDF file: {error_msg}")


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
        except Exception:
            raise ValueError(
                f"Unsupported file type: {filename}. "
                f"Supported types: PDF (.pdf), DOCX (.docx), TXT (.txt)"
            )

