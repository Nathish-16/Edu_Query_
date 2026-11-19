import os
from typing import List, Dict, Any
from pptx import Presentation
from backend.rag.chunking import chunk_text, chunk_slide

# Robust PyMuPDF import
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def extract_pdf_chunks(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text from PDF page-by-page and chunks each page.
    Handles scanned/empty pages and whitespace normalization cleanly.
    """
    chunks = []
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            # If plain text extraction yields nothing, attempt block extraction
            if not text:
                blocks = page.get_text("blocks")
                text = " ".join([b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()]).strip()
            
            if not text:
                continue
                
            # Base metadata for the entire page
            base_metadata = {
                "source": file_name,
                "type": "pdf",
                "page_number": page_num + 1,
                "total_pages": total_pages
            }
            
            # Chunk the page text (using sliding window if it's large)
            page_chunks = chunk_text(text, chunk_size=800, chunk_overlap=150, metadata=base_metadata)
            chunks.extend(page_chunks)
            
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
        raise e
        
    return chunks


def _extract_shape_elements(shape, title_id=None) -> List[str]:
    """Recursively extract text and table content from shapes and group shapes."""
    elements = []
    
    # Skip title shape if already processed
    if title_id is not None and getattr(shape, "shape_id", None) == title_id:
        return elements
        
    # 1. Text frames
    if getattr(shape, "has_text_frame", False):
        try:
            for paragraph in shape.text_frame.paragraphs:
                p_text = paragraph.text.strip()
                if p_text:
                    elements.append(p_text)
        except Exception:
            pass

    # 2. Tables
    if getattr(shape, "has_table", False):
        try:
            table_rows = []
            for row in shape.table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                elements.append("Table:\n" + "\n".join(table_rows))
        except Exception:
            pass

    # 3. Group shapes (recursive)
    if hasattr(shape, "shapes"):
        try:
            for sub_shape in shape.shapes:
                elements.extend(_extract_shape_elements(sub_shape, title_id=title_id))
        except Exception:
            pass
            
    return elements


def extract_pptx_chunks(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text, tables, group shapes, and speaker notes from PPTX slide-by-slide.
    """
    chunks = []
    try:
        prs = Presentation(file_path)
        total_slides = len(prs.slides)
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_elements = []
            slide_title = ""
            title_id = None
            
            # 1. Try to find the slide title
            try:
                if slide.shapes.title and hasattr(slide.shapes.title, "text"):
                    slide_title = slide.shapes.title.text.strip()
                    title_id = getattr(slide.shapes.title, "shape_id", None)
                    if slide_title:
                        slide_elements.append(f"Title: {slide_title}")
            except Exception:
                pass
            
            # 2. Extract text from shapes (including tables & group shapes)
            for shape in slide.shapes:
                slide_elements.extend(_extract_shape_elements(shape, title_id=title_id))
            
            # 3. Extract speaker notes
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        slide_elements.append(f"Speaker Notes: {notes_text}")
            except Exception:
                pass
            
            # Combine all slide text
            slide_content = "\n".join(slide_elements).strip()
            
            if not slide_content:
                continue
                
            base_meta = {
                "source": file_name,
                "type": "pptx",
                "slide_number": slide_num,
                "slide_count": total_slides,
                "total_pages": total_slides,
                "title": slide_title or f"Slide {slide_num}"
            }
            
            # If slide text is large (>1000 chars), chunk it to fit embedding model nicely
            if len(slide_content) > 1000:
                slide_chunks = chunk_text(slide_content, chunk_size=800, chunk_overlap=150, metadata=base_meta)
                chunks.extend(slide_chunks)
            else:
                chunk = chunk_slide(
                    slide_text=slide_content,
                    slide_num=slide_num,
                    title=slide_title,
                    source=file_name,
                    total_slides=total_slides
                )
                chunks.append(chunk)
            
    except Exception as e:
        print(f"Error reading PPTX {file_name}: {e}")
        raise e
        
    return chunks


def ingest_file(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Ingests any supported file and returns list of chunks.
    
    Args:
        file_path (str): The full system path to the file.
        file_name (str): The file name for document metadata.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing 'text' and 'metadata'.
    """
    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == '.pdf':
        return extract_pdf_chunks(file_path, file_name)
    elif ext == '.pptx':
        return extract_pptx_chunks(file_path, file_name)
    elif ext == '.ppt':
        # python-pptx only parses modern OpenXML presentations (.pptx).
        try:
            return extract_pptx_chunks(file_path, file_name)
        except Exception:
            raise ValueError(
                f"Legacy binary format '{file_name}' cannot be read directly. "
                "Please save or export as modern PowerPoint presentation (.pptx) or PDF."
            )
    else:
        # Fallback for plain text files (.txt, .md, etc.)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            base_metadata = {
                "source": file_name,
                "type": "text",
                "total_pages": 1
            }
            return chunk_text(text, chunk_size=800, chunk_overlap=150, metadata=base_metadata)
        except Exception as e:
            print(f"Error reading text file {file_name}: {e}")
            raise ValueError(f"Unsupported file format: {ext}")
