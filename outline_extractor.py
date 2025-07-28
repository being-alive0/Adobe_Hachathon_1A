# outline_extractor.py

import fitz  # PyMuPDF
import json
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

def parse_pdf_enhanced(file_bytes: bytes) -> Dict[int, List[Dict[str, Any]]]:
    """Parses a PDF, extracting all necessary raw data including font family and bold status."""
    pages_content = {}
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
        page_lines = []
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join([span['text'] for span in line['spans']]).strip()
                    if not line_text or len(line['spans']) == 0: continue
                    
                    first_span = line['spans'][0]
                    font_family = first_span['font'].split('-')[0]
                    is_bold = "bold" in first_span['font'].lower()
                    
                    page_lines.append({
                        "text": line_text,
                        "size": round(first_span['size']),
                        "font_family": font_family,
                        "bold": is_bold,
                        "page_num": page_num,
                        "bbox": line['bbox']
                    })
        pages_content[page_num] = sorted(page_lines, key=lambda x: x['bbox'][1])
    return pages_content

def find_and_parse_toc(pages_content: Dict[int, List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Finds and parses a Table of Contents, returning a structured list with correct page numbers."""
    TOC_SYNONYMS = ['table of contents', 'contents', 'toc', 'outline', 'index', 'summary', 'document map']
    toc_entries = []
    max_page_to_scan = int(len(pages_content) * 0.2) + 3

    for page_num in range(min(max_page_to_scan, len(pages_content))):
        lines = pages_content.get(page_num, [])
        if not lines: continue
        
        if any(line['text'].lower().strip() in TOC_SYNONYMS for line in lines[:3]):
            for p_num in range(page_num, min(page_num + 5, len(pages_content))):
                for toc_line in pages_content.get(p_num, []):
                    match = re.match(r'^(.*?)(?:\s*\.|\s){2,}(\d+)$', toc_line['text'])
                    if match:
                        text = match.group(1).strip()
                        text = re.sub(r'^\d+[\.\s]*', '', text)
                        page = int(match.group(2)) - 1
                        indent = toc_line['bbox'][0]
                        toc_entries.append({'text': text, 'page': page, 'indent': indent})
            
            if toc_entries:
                indents = sorted(list(set([e['indent'] for e in toc_entries])))
                indent_map = {indent: f"H{i+1}" for i, indent in enumerate(indents)}
                for entry in toc_entries:
                    entry['level'] = indent_map.get(entry['indent'], 'H1')
                return toc_entries
    return None

def engineer_features_layout(pages_content: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Engineers features based on layout and style for the ML fallback model."""
    all_lines = [line for page_num in sorted(pages_content.keys()) for line in pages_content[page_num]]
    if not all_lines: return []

    font_families = sorted(list(set([line['font_family'] for line in all_lines])))
    font_map = {font: i for i, font in enumerate(font_families)}

    for line in all_lines:
        bbox = line['bbox']
        page_width = pages_content[line['page_num']][0]['bbox'][2] if pages_content[line['page_num']] else 1
        
        line['pattern_match'] = 1 if re.match(r"^(?:(\d{1,2}(?:\.\d{1,2})*))", line['text']) else 0
        line['is_centered'] = 1 if 0.4 < ((bbox[0] + bbox[2]) / 2 / page_width) < 0.6 else 0
        line['indentation'] = bbox[0]
        line['font_family_encoded'] = font_map.get(line['font_family'], -1)
        
    return all_lines

def stitch_multiline_headings(headings: List[Dict], all_lines_map: Dict[int, Dict[Tuple, Dict]]) -> List[Dict]:
    """Stitches headings that span multiple lines."""
    if not headings: return []
    
    stitched_headings = []
    processed_bboxes = set()

    for i, h in enumerate(headings):
        current_bbox = tuple(h['bbox'])
        if (h['page_num'], current_bbox) in processed_bboxes:
            continue

        merged_text = h['text']
        
        page_lines = all_lines_map.get(h['page_num'], {})
        sorted_page_lines = sorted(page_lines.values(), key=lambda x: x['bbox'][1])
        
        try:
            current_line_index = next(i for i, line in enumerate(sorted_page_lines) if tuple(line['bbox']) == current_bbox)
        except StopIteration:
            stitched_headings.append(h)
            continue

        j = current_line_index + 1
        while j < len(sorted_page_lines):
            next_line = sorted_page_lines[j]
            is_potential_heading = next_line.get('is_heading_prediction', 0) == 1
            if (len(next_line['text'].split()) < 7 and 
                not next_line['bold'] and 
                abs(next_line['size'] - h['size']) <= 1 and
                not is_potential_heading):
                merged_text += " " + next_line['text']
                processed_bboxes.add((next_line['page_num'], tuple(next_line['bbox'])))
                j += 1
            else:
                break
        
        h['text'] = merged_text
        stitched_headings.append(h)
        
    return stitched_headings

def nlp_sanity_check(headings: List[Dict]) -> List[Dict]:
    """A final lightweight filter to remove gibberish or noise from the output."""
    clean_headings = []
    for h in headings:
        text = h['text']
        if (re.search(r'[a-zA-Z]', text) and 
            '```' not in text and 
            len(text.split()) < 25):
            clean_headings.append(h)
    return clean_headings

def assign_levels_by_font_size(headings: List[Dict]) -> List[Dict]:
    """Assigns H1, H2, H3 levels based on font size."""
    if not headings: return []
    sizes = sorted(list(set([h['size'] for h in headings])), reverse=True)
    size_map = {size: f"H{i+1}" for i, size in enumerate(sizes[:3])}
    for h in headings:
        h['level'] = size_map.get(h['size'], 'H3')
    return headings

def process_document_definitive(file_bytes: bytes, model: Optional[lgb.Booster] = None) -> str:
    """The definitive pipeline using the hybrid ToC-first strategy."""
    pages_content = parse_pdf_enhanced(file_bytes)
    
    parsed_toc = find_and_parse_toc(pages_content)
    if parsed_toc and len(parsed_toc) > 4:
        print("INFO: Substantial Table of Contents found. Using ToC as primary source.")
        page_0_lines = pages_content.get(0, [])
        title_text = " ".join([l['text'] for l in page_0_lines if l['size'] == max([p['size'] for p in page_0_lines])]) if page_0_lines else "Untitled"
        
        outline = [{'level': e['level'], 'text': e['text'], 'page': e['page']} for e in parsed_toc]
        return json.dumps({"title": title_text, "outline": outline}, indent=2)

    print("INFO: ToC not found or not substantial. Using ML inference model.")
    if not model:
        return json.dumps({"title": "Error: ML Model not provided for inference.", "outline": []}, indent=2)

    lines_with_features = engineer_features_layout(pages_content)
    if not lines_with_features:
        return json.dumps({"title": "Could not parse document", "outline": []}, indent=2)
        
    df = pd.DataFrame(lines_with_features)
    features_for_model = ['size', 'pattern_match', 'bold', 'is_centered', 'indentation', 'font_family_encoded']
    for col in features_for_model:
        if col not in df.columns: df[col] = 0

    predictions = model.predict(df[features_for_model])
    df['is_heading_prediction'] = (predictions > 0.5).astype(int)
    
    page_0_lines = pages_content.get(0, [])
    title_text = " ".join([l['text'] for l in page_0_lines if l['size'] == max([p['size'] for p in page_0_lines])]) if page_0_lines else "Untitled"

    all_lines_map = {pageNum: {tuple(line['bbox']): line for line in lines} for pageNum, lines in pages_content.items()}
    
    heading_candidates = df[df['is_heading_prediction'] == 1].to_dict('records')
    stitched_headings = stitch_multiline_headings(heading_candidates, all_lines_map)
    
    final_outline_with_levels = assign_levels_by_font_size(stitched_headings)
    final_clean_outline = nlp_sanity_check(final_outline_with_levels)
            
    final_output = []
    for h in final_clean_outline:
        final_output.append({
            'level': h['level'],
            'text': h['text'],
            'page': h['page_num']
        })

    result = {"title": title_text, "outline": sorted(final_output, key=lambda x: x['page'])}
    return json.dumps(result, indent=2)