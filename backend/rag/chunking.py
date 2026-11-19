import re
from typing import List, Dict, Any

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Chunks a continuous string of text using a sliding window.
    Attaches the provided base metadata to each chunk.
    
    Args:
        text (str): The raw text to chunk.
        chunk_size (int): Max character length of a single chunk.
        chunk_overlap (int): Overlap size in characters.
        metadata (dict): Base metadata dict to attach to each chunk (e.g. source, page).
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing 'text' and 'metadata'.
    """
    if metadata is None:
        metadata = {}
        
    # Clean whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
        
    chunks = []
    
    # If the text is shorter than chunk_size, just return it as a single chunk
    if len(text) <= chunk_size:
        chunks.append({
            "text": text,
            "metadata": metadata.copy()
        })
        return chunks
        
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a clean boundary (space or sentence end) near the end of the chunk
        if end < len(text):
            # Look backward for a sentence boundary or space to avoid cutting words/sentences
            boundary = -1
            for offset in range(0, min(100, chunk_overlap)):
                char = text[end - offset]
                if char in ['.', '!', '?']:
                    boundary = end - offset + 1
                    break
                elif char == ' ' and boundary == -1:
                    boundary = end - offset
            
            if boundary != -1:
                end = boundary
                
        chunk_content = text[start:end].strip()
        if chunk_content:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            
            chunks.append({
                "text": chunk_content,
                "metadata": chunk_metadata
            })
            chunk_idx += 1
            
        # Move the sliding window
        start = end - chunk_overlap
        if start >= len(text) or (end >= len(text)):
            break
            
    return chunks

def chunk_slide(
    slide_text: str,
    slide_num: int,
    title: str,
    source: str,
    total_slides: int = None
) -> Dict[str, Any]:
    """
    Formats slide-level text as a single discrete chunk with slide-specific metadata.
    
    Args:
        slide_text (str): Extracted text from all shapes in the slide.
        slide_num (int): The 1-based slide number.
        title (str): Slide title.
        source (str): Document source file name.
        total_slides (int, optional): Total number of slides in the presentation.
        
    Returns:
        Dict[str, Any]: Chunk data dictionary.
    """
    # Clean whitespace
    clean_text = re.sub(r'\s+', ' ', slide_text).strip()
    
    slide_total = total_slides or slide_num
    metadata = {
        "source": source,
        "type": "pptx",
        "slide_number": slide_num,
        "slide_count": slide_total,
        "total_pages": slide_total,
        "title": title or f"Slide {slide_num}"
    }
    
    return {
        "text": clean_text,
        "metadata": metadata
    }
