import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from settings import MIN_CHARS_PER_CHUNK

import pymupdf
import uuid
import datetime
import re
import os
from copy import deepcopy
import json

# Update paths to use absolute paths from project root
pdf_path = str(project_root / "data" / "pdfs")
path_save = str(project_root / "data" / "segments" / "segments.json")

def id_generator():
    return str(uuid.uuid4())

def process_page(page):
    page_text = page.get_text(sort=True)

    # Remove trailing whitespace
    page_text = page_text.strip()

    # Remove page number in the beginning
    if page_text.startswith("Seite"):
        # remove "Seite " until the next newline
        page_text = page_text[page_text.find("\n")+1:].strip()

    # Remove double whitespace
    while "  " in page_text:
        page_text = page_text.replace("  ", " ")

    return page_text

def starts_with_number(text):
    return text[0].isdigit()

# filenames are in format 'BMF_2023_10_05.pdf'
def date_from_filename(filename):
    year = filename.split("_")[1]
    month = filename.split("_")[2]
    day = filename.split("_")[3].split(".")[0]
    return datetime.datetime(int(year), int(month), int(day))

# A function that checks if all segments have valid next_ids and previous_ids
def test_segment_connections(segments):
    # Convert to set for O(1) lookup
    all_ids = {segment["id"] for segment in segments}
    
    # Check next_ids
    missing_next_ids = []
    for segment in segments:
        if segment["next_id"] is not None and segment["next_id"] not in all_ids:
            missing_next_ids.append({
                "segment_id": segment["id"],
                "missing_next_id": segment["next_id"],
                "page": segment["page"],
                "filename": segment["filename"]
            })
    
    # Check previous_ids
    missing_prev_ids = []
    for segment in segments:
        if segment["previous_id"] is not None and segment["previous_id"] not in all_ids:
            missing_prev_ids.append({
                "segment_id": segment["id"],
                "missing_previous_id": segment["previous_id"],
                "page": segment["page"],
                "filename": segment["filename"]
            })
    
    # Print results
    if missing_next_ids:
        print(f"\nFound {len(missing_next_ids)} segments with non-existent next_ids:")
        for error in missing_next_ids:
            print(f"- Segment {error['segment_id']} (page {error['page']} in {error['filename']}) "
                  f"points to non-existent next_id: {error['missing_next_id']}")
    
    if missing_prev_ids:
        print(f"\nFound {len(missing_prev_ids)} segments with non-existent previous_ids:")
        for error in missing_prev_ids:
            print(f"- Segment {error['segment_id']} (page {error['page']} in {error['filename']}) "
                  f"points to non-existent previous_id: {error['missing_previous_id']}")
    
    if not missing_next_ids and not missing_prev_ids:
        print("All segment connections are valid!")
    
    return len(missing_next_ids) == 0 and len(missing_prev_ids) == 0

def is_heading(text: str, max_chars=100) -> bool:
    """
    Determines if the given text is likely a heading. 
    A heading in this context:
      - Starts with one or more letters or digits (e.g., '1', '398', 'a', 'aa', etc.).
      - Is immediately followed by either a dot (.) or a closing parenthesis () ), 
        optionally followed by a space or end of string.
    Examples:
      "1."   -> True
      "5. Some Heading" -> True
      "b) Another Heading" -> True
      "398) Text" -> True
      "Heading" -> False
      "1)Something" -> True (though unusual, pattern still matches)
    """
    if len(text.strip()) > max_chars:
        # if the text is too long, it is not a heading
        return False
    
    pattern = r'^[0-9A-Za-z]+[.)](\s|$)'
    return bool(re.match(pattern, text.strip()))



segments = []
for filename in os.listdir(pdf_path):
    print(f"Processing {filename}")
    document_date = date_from_filename(filename).strftime("%d.%m.%Y")
    full_path = os.path.join(pdf_path, filename)
    pdf_document = pymupdf.open(full_path)
    next_id = id_generator()
    last_id = None
    chunk_memory = ""
    for i_page, page in enumerate(pdf_document):
        text = process_page(page)
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > MIN_CHARS_PER_CHUNK]
        for i_chunk, chunk in enumerate(chunks):
            # Check if this chunk belongs to the last chunk due to a line break.
            if i_chunk == 0 and i_page > 0 and starts_with_number(segments[-1]["text"]) and not starts_with_number(chunk):
                # this chunk belongs to the last chunk
                segments[-1]["text"] += "\n\n" + chunk
                segments[-1]["page"] += f" und {i_page+1}"
                continue

            if is_heading(chunk):
                # this chunk is a heading
                # Save the chunk for the next chunk
                chunk_memory += "\n\n" + chunk
                chunk_memory = chunk_memory.strip()
                continue
                
            # Assign correct IDs to each segment
            current_id = deepcopy(next_id) if next_id is not None else id_generator()
            next_id = None if (i_page == len(pdf_document) - 1 and i_chunk == len(chunks) - 1) else id_generator()

            # Assemble a chunk if there is a chunk_memory (e.g. from a previous heading)
            if chunk_memory:
                chunk = chunk_memory + "\n\n" + chunk
                chunk_memory = ""
            
            segments.append({
                "id": current_id,
                "full_path": full_path,
                "filename": filename,
                "page": str(i_page+1),
                "text": chunk,
                "previous_id": last_id,
                "next_id": next_id,
                "document_date": document_date
            })
            
            last_id = deepcopy(current_id)

print(f"Found {len(segments)} segments from {len(os.listdir(pdf_path))} PDFs")
            
with open(path_save, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False)

print(f"Saved segments to {path_save}")
print(f"Script finished")