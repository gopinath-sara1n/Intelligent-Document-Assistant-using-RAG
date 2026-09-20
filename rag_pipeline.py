"""
rag_pipeline.py
=============================================================================
DocuMind Core RAG Engine
Implements the end-to-end multimodal RAG pipeline:
  1. PDF Ingestion & Extraction (Text, Tables, Images via Docling / Fallback)
  2. Gemini Multimodal Visual Understanding (Charts, Diagrams, Figures)
  3. Canonical Document Representation
  4. Structure-Aware Chunking (Intact tables, intact visual descriptions)
  5. Jina v4 Embeddings (2048-dim normalized vectors)
  6. FAISS Vector Database (IndexFlatIP Cosine Similarity)
  7. Semantic Retrieval, Exact Keyword Search & Grounded Gemini Generation
=============================================================================
"""

import os
import re
import json
import time
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from PIL import Image
import io
import requests
import faiss
from pypdf import PdfReader
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 0. ENVIRONMENT & CONFIGURATION
# ---------------------------------------------------------------------------

def load_env_vars():
    """Load API keys from .env file or environment variables."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_vars()

# Jina Embedding API Configuration
JINA_EMBEDDING_MODEL_NAME = "jina-embeddings-v4"
JINA_EMBEDDING_OUTPUT_DIMENSIONALITY = 2048
JINA_EMBEDDING_ENDPOINT = "https://api.jina.ai/v1/embeddings"
JINA_EMBEDDING_RPM_LIMIT = 500
JINA_EMBEDDING_SAFE_RPM = 400
JINA_EMBEDDING_BATCH_SIZE = 50
JINA_EMBEDDING_INTER_BATCH_DELAY = 0.5
JINA_EMBEDDING_MIN_REQUEST_INTERVAL = 0.15
JINA_EMBEDDING_MAX_RETRIES = 5

# Gemini Vision & Answer Generation Configuration
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash"
]

# Chunking Configuration
TARGET_CHARS = 1800
MAX_CHARS = 3200
MIN_STANDALONE_CHARS = 150

# Visual Extraction Limits
VISUAL_MIN_WIDTH = 150
VISUAL_MIN_HEIGHT = 80


@dataclass
class DocumentStats:
    total_pages: int = 0
    total_chunks: int = 0
    total_tables: int = 0
    total_pictures: int = 0
    total_elements: int = 0
    embedding_dim: int = 2048
    processing_time_sec: float = 0.0


@dataclass
class DocuMindIndex:
    chunks: List[Dict[str, Any]]
    metadata: List[Dict[str, Any]]
    faiss_index: Any
    stats: DocumentStats
    visual_manifest: List[Dict[str, Any]] = field(default_factory=list)
    canonical_elements: List[Dict[str, Any]] = field(default_factory=list)
    pdf_path: Optional[str] = None
    pdf_name: str = ""


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING & HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def clean_text(text: Any) -> str:
    """Normalize whitespace and remove excessive newlines/special characters."""
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences preserving structural boundaries."""
    text = clean_text(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z₹$0-9(\[])", paragraph)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def is_probable_page_number(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if re.fullmatch(r"\d{1,4}", text):
        return True
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,8}", text):
        return True
    return False


def is_low_value_fragment(text: str) -> bool:
    text = clean_text(text)
    if not text or is_probable_page_number(text):
        return True
    return len(text) < MIN_STANDALONE_CHARS


# ---------------------------------------------------------------------------
# 2. DOCUMENT PARSING (DOCLING WITH ROBUST NATIVE FALLBACK)
# ---------------------------------------------------------------------------

def _parse_with_docling(pdf_path: str, work_dir: Path, progress_cb: Optional[Callable] = None):
    """Parse document using Docling DocumentConverter."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if progress_cb:
        progress_cb(15, "Running Docling document parser...")

    result = converter.convert(str(pdf_path))
    doc = result.document
    docling_data = doc.export_to_dict()

    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    picture_manifest = []
    for idx, picture in enumerate(doc.pictures, start=1):
        picture_id = f"picture_{idx:03d}"
        image_path = image_dir / f"{picture_id}.png"
        try:
            image = picture.get_image(doc)
            if image is None:
                continue
            image.save(image_path)
            page_no = None
            prov = getattr(picture, "prov", [])
            if prov and len(prov) > 0:
                first_prov = prov[0]
                page_no = getattr(first_prov, "page_no", None) or (first_prov.get("page_no") if isinstance(first_prov, dict) else None)
            
            picture_manifest.append({
                "picture_id": picture_id,
                "index": idx - 1,
                "page": page_no,
                "image_path": str(image_path),
                "width": image.width,
                "height": image.height,
                "bbox_width": image.width,
                "bbox_height": image.height
            })
        except Exception:
            pass

    return docling_data, picture_manifest


def _parse_with_pdfplumber(pdf_path: str, work_dir: Path, progress_cb: Optional[Callable] = None):
    """High-fidelity fallback parser using pdfplumber & pypdfium2."""
    import pdfplumber
    import pypdfium2

    if progress_cb:
        progress_cb(15, "Parsing document structure, tables, and figures...")

    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    texts_list = []
    tables_list = []
    pictures_list = []
    body_children = []

    pdf_doc = pypdfium2.PdfDocument(pdf_path)
    total_pages = len(pdf_doc)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            if progress_cb and total_pages > 0:
                pct = 15 + int((page_idx / total_pages) * 20)
                progress_cb(pct, f"Extracting page {page_idx} of {total_pages}...")

            # 1. Extract tables
            try:
                tables = page.extract_tables()
                for t_idx, table_grid in enumerate(tables):
                    if not table_grid or len(table_grid) == 0:
                        continue
                    df = pd.DataFrame(table_grid[1:], columns=table_grid[0]) if len(table_grid) > 1 else pd.DataFrame(table_grid)
                    df = df.fillna("")
                    # Convert to markdown formatted string
                    lines = []
                    cols = [clean_text(c) for c in df.columns]
                    if any(cols):
                        lines.append(" | ".join(cols))
                    for _, row in df.iterrows():
                        vals = [clean_text(v) for v in row.tolist()]
                        if any(vals):
                            lines.append(" | ".join(v if v else "-" for v in vals))
                    table_text = "\n".join(lines).strip()
                    if table_text:
                        ref_id = f"table_{page_idx}_{t_idx}"
                        table_item = {
                            "self_ref": ref_id,
                            "text": table_text,
                            "prov": [{"page_no": page_idx}]
                        }
                        tables_list.append(table_item)
                        body_children.append({"$ref": ref_id})
            except Exception:
                pass

            # 2. Extract text paragraphs
            try:
                text_content = page.extract_text()
                if text_content:
                    paragraphs = text_content.split("\n\n")
                    for p_idx, para in enumerate(paragraphs):
                        cleaned = clean_text(para)
                        if not cleaned:
                            continue
                        ref_id = f"text_{page_idx}_{p_idx}"
                        # Check if paragraph looks like a section header
                        is_header = len(cleaned) < 120 and (cleaned.isupper() or re.match(r"^(Section|\d+(\.\d+)*)\s+", cleaned))
                        text_item = {
                            "self_ref": ref_id,
                            "text": cleaned,
                            "label": "section_header" if is_header else "text",
                            "prov": [{"page_no": page_idx}]
                        }
                        texts_list.append(text_item)
                        body_children.append({"$ref": ref_id})
            except Exception:
                pass

            # 3. Extract pictures from page via pypdf
            try:
                reader = PdfReader(pdf_path)
                pypdf_page = reader.pages[page_idx - 1]
                for img_idx, img in enumerate(pypdf_page.images, start=1):
                    try:
                        import io
                        pil_img = Image.open(io.BytesIO(img.data))
                        w, h = pil_img.size
                        if w >= VISUAL_MIN_WIDTH and h >= VISUAL_MIN_HEIGHT:
                            pic_id = f"picture_{len(pictures_list)+1:03d}"
                            img_path = image_dir / f"{pic_id}.png"
                            if pil_img.mode not in ("RGB", "RGBA"):
                                pil_img = pil_img.convert("RGB")
                            pil_img.save(img_path, format="PNG")
                            ref_id = f"pic_{pic_id}"
                            pic_item = {
                                "self_ref": ref_id,
                                "picture_id": pic_id,
                                "image_path": str(img_path),
                                "width": w,
                                "height": h,
                                "bbox_width": w,
                                "bbox_height": h,
                                "page": page_idx,
                                "prov": [{"page_no": page_idx}]
                            }
                            pictures_list.append(pic_item)
                            body_children.append({"$ref": ref_id})
                    except Exception:
                        pass
            except Exception:
                pass

    docling_data = {
        "name": Path(pdf_path).name,
        "texts": texts_list,
        "tables": tables_list,
        "pictures": pictures_list,
        "body": {"children": body_children}
    }
    return docling_data, pictures_list


def parse_pdf(pdf_path: str, work_dir: Path, progress_cb: Optional[Callable] = None):
    """Main document parsing entrypoint with intelligent fallback."""
    try:
        import docling
        return _parse_with_docling(pdf_path, work_dir, progress_cb)
    except Exception:
        # Graceful fallback to pdfplumber + pypdfium2
        return _parse_with_pdfplumber(pdf_path, work_dir, progress_cb)


# ---------------------------------------------------------------------------
# 3. GEMINI MULTIMODAL VISUAL UNDERSTANDING
# ---------------------------------------------------------------------------

def build_visual_prompt(page_number: Optional[int] = None) -> str:
    pg_str = f"page {page_number}" if page_number else "the document"
    return f"""
You are analyzing a visual extracted from {pg_str} of a complex PDF document for a Retrieval-Augmented Generation (RAG) system.

Create a highly accurate, self-contained description of the visual.

Include, when visible:
1. Chart or figure title.
2. Type of visual (bar chart, flowchart, schematic diagram, photograph, architecture, table, etc.).
3. Important categories, labels and legends.
4. Important numerical values and statistics.
5. Dates, years or periods.
6. Units of measurement.
7. Trends, comparisons, correlations and relationships.
8. Forecast values, if present.
9. Important annotations or callouts.
10. Source, if visible.
11. Meaning or interpretation directly supported by the visual.

Do NOT invent information that is not visible.
Do NOT provide opinions or recommendations.
Preserve exact terminology, names, labels and numerical values whenever readable.

Return ONLY the description.
""".strip()


def describe_visuals_with_gemini(
    visual_manifest: List[Dict[str, Any]],
    gemini_api_key: str,
    progress_cb: Optional[Callable] = None
) -> Dict[str, Dict[str, Any]]:
    """Analyze extracted images using Gemini Multimodal Vision with fallback."""
    if not visual_manifest or not gemini_api_key:
        return {}

    client = genai.Client(api_key=gemini_api_key)
    results_by_picture = {}
    total_visuals = len(visual_manifest)

    for idx, item in enumerate(visual_manifest, start=1):
        pic_id = item.get("picture_id", f"pic_{idx}")
        page_no = item.get("page")
        image_path = item.get("image_path")

        if progress_cb:
            pct = 35 + int((idx / max(total_visuals, 1)) * 25)
            progress_cb(pct, f"Analyzing visual {idx}/{total_visuals} ({pic_id}) with Gemini Vision...")

        if not image_path or not Path(image_path).exists():
            continue

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception:
            continue

        prompt = build_visual_prompt(page_no)
        description = None
        model_used = None

        mime_type = "image/jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "image/png"

        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt
                    ]
                )
                if response.text and response.text.strip():
                    description = response.text.strip()
                    model_used = model_name
                    break
            except Exception as exc:
                err_str = str(exc).upper()
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(2.0)
                    continue
                continue

        if description:
            results_by_picture[pic_id] = {
                "picture_id": pic_id,
                "page": page_no,
                "image_path": image_path,
                "model": model_used,
                "description": description,
                "status": "success"
            }
            time.sleep(0.5)
        else:
            results_by_picture[pic_id] = {
                "picture_id": pic_id,
                "page": page_no,
                "image_path": image_path,
                "description": None,
                "status": "failed"
            }

    return results_by_picture


# ---------------------------------------------------------------------------
# 4. CANONICAL DOCUMENT ASSEMBLY
# ---------------------------------------------------------------------------

def build_canonical_document(
    docling_data: Dict[str, Any],
    gemini_visuals: Dict[str, Dict[str, Any]],
    picture_manifest: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merges texts, tables, and visual descriptions in document reading order."""
    texts_by_ref = {item.get("self_ref"): item for item in docling_data.get("texts", []) if item.get("self_ref")}
    tables_by_ref = {item.get("self_ref"): item for item in docling_data.get("tables", []) if item.get("self_ref")}
    pictures_by_ref = {item.get("self_ref"): item for item in docling_data.get("pictures", []) if item.get("self_ref")}

    picture_ref_to_id = {}
    for idx, pic in enumerate(picture_manifest, start=1):
        s_ref = pic.get("self_ref")
        p_id = pic.get("picture_id", f"picture_{idx:03d}")
        if s_ref:
            picture_ref_to_id[s_ref] = p_id

    canonical_elements = []
    body_children = docling_data.get("body", {}).get("children", [])

    for position, child in enumerate(body_children):
        ref = child.get("$ref") if isinstance(child, dict) else str(child)
        if not ref:
            continue

        # 1. TEXT
        if ref in texts_by_ref:
            item = texts_by_ref[ref]
            text = clean_text(item.get("text") or item.get("content"))
            if not text:
                continue
            prov = item.get("prov", [])
            page_no = prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}",
                "type": "text",
                "position": position,
                "page": page_no,
                "text": text,
                "label": item.get("label", "text"),
                "source": {"ref": ref}
            })

        # 2. TABLE
        elif ref in tables_by_ref:
            item = tables_by_ref[ref]
            table_text = clean_text(item.get("text") or item.get("content"))
            if not table_text:
                continue
            prov = item.get("prov", [])
            page_no = prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}",
                "type": "table",
                "position": position,
                "page": page_no,
                "text": table_text,
                "content": table_text,
                "source": {"ref": ref}
            })

        # 3. PICTURE / VISUAL
        elif ref in pictures_by_ref or ref in picture_ref_to_id:
            item = pictures_by_ref.get(ref, {})
            p_id = picture_ref_to_id.get(ref) or item.get("picture_id")
            gemini_res = gemini_visuals.get(p_id)
            if not gemini_res or not gemini_res.get("description"):
                continue
            desc = gemini_res["description"].strip()
            prov = item.get("prov", [])
            page_no = item.get("page") or (prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None)
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}",
                "type": "visual",
                "position": position,
                "page": page_no,
                "picture_id": p_id,
                "image_path": gemini_res.get("image_path"),
                "visual_description": desc,
                "text": desc,
                "content": desc,
                "source": {"ref": ref, "gemini_model": gemini_res.get("model")}
            })

    # Ensure all pictures with valid Gemini descriptions are present in canonical_elements
    processed_picture_ids = {el.get("picture_id") for el in canonical_elements if el.get("type") == "visual"}
    for pic in picture_manifest:
        p_id = pic.get("picture_id")
        if p_id and p_id not in processed_picture_ids:
            gemini_res = gemini_visuals.get(p_id)
            if gemini_res and gemini_res.get("description"):
                desc = gemini_res["description"].strip()
                prov = pic.get("prov", [])
                page_no = pic.get("page") or (prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None)
                canonical_elements.append({
                    "element_id": f"element_{len(canonical_elements)+1:06d}",
                    "type": "visual",
                    "position": len(canonical_elements),
                    "page": page_no,
                    "picture_id": p_id,
                    "image_path": gemini_res.get("image_path"),
                    "visual_description": desc,
                    "text": desc,
                    "content": desc,
                    "source": {"ref": pic.get("self_ref", p_id), "gemini_model": gemini_res.get("model")}
                })

    return canonical_elements


# ---------------------------------------------------------------------------
# 5. STRUCTURE-AWARE CHUNKING
# ---------------------------------------------------------------------------

def create_structure_aware_chunks(canonical_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Structure-Aware Chunking:
      - Tables kept as intact standalone chunks
      - Visual descriptions kept as intact standalone chunks
      - Section headings preserved as retrieval context
      - Text split on sentences within [TARGET_CHARS, MAX_CHARS]
      - Low-value fragments merged
    """
    REMOVE_LABELS = {"page_header", "page_footer"}
    chunking_elements = [
        el for el in canonical_elements
        if not (el["type"] == "text" and el.get("label", "text") in REMOVE_LABELS)
    ]

    chunks = []
    current_parts = []
    current_element_ids = []
    current_element_types = []
    current_pages = []
    current_section = ""

    def get_current_text():
        return "\n\n".join(x for x in current_parts if x.strip()).strip()

    def flush_chunk():
        nonlocal current_parts, current_element_ids, current_element_types, current_pages
        text = get_current_text()
        if not text:
            current_parts = []
            current_element_ids = []
            current_element_types = []
            current_pages = []
            return

        chunks.append({
            "chunk_id": f"chunk_{len(chunks):06d}",
            "page_start": min(current_pages) if current_pages else None,
            "page_end": max(current_pages) if current_pages else None,
            "section": current_section,
            "content": text,
            "element_ids": current_element_ids.copy(),
            "element_types": current_element_types.copy(),
            "has_table": "table" in current_element_types,
            "has_visual": "visual" in current_element_types
        })
        current_parts = []
        current_element_ids = []
        current_element_types = []
        current_pages = []

    def add_element(element, content):
        content = clean_text(content)
        if not content:
            return
        current_parts.append(content)
        current_element_ids.append(element["element_id"])
        current_element_types.append(element["type"])
        if element.get("page") is not None:
            current_pages.append(element["page"])

    def add_text_sentence(element, sentence):
        candidate = get_current_text()
        if candidate:
            candidate = candidate + "\n\n" + sentence
        else:
            candidate = sentence

        if len(candidate) <= TARGET_CHARS:
            add_element(element, sentence)
            return

        if get_current_text():
            flush_chunk()
            if len(sentence) <= MAX_CHARS:
                add_element(element, sentence)
                return

        # Sentence itself exceeds MAX_CHARS: split on words
        words = sentence.split()
        buffer = []
        for word in words:
            test = " ".join(buffer + [word])
            if len(test) <= MAX_CHARS:
                buffer.append(word)
            else:
                if buffer:
                    add_element(element, " ".join(buffer))
                    flush_chunk()
                buffer = [word]
        if buffer:
            add_element(element, " ".join(buffer))

    for element in chunking_elements:
        el_type = element["type"]
        label = element.get("label", "")
        page = element.get("page")

        # Section Header
        if el_type == "text" and label == "section_header":
            header = clean_text(element.get("text", ""))
            if not header:
                continue
            flush_chunk()
            current_section = header
            add_element(element, header)
            continue

        # Table: keep intact as standalone chunk
        if el_type == "table":
            flush_chunk()
            table_text = clean_text(element.get("content") or element.get("text") or "")
            if table_text:
                add_element(element, table_text)
                flush_chunk()
            continue

        # Visual description: keep intact as standalone chunk
        if el_type == "visual":
            flush_chunk()
            desc = clean_text(element.get("visual_description") or element.get("content") or "")
            if not desc:
                continue
            visual_text = (
                f"[VISUAL]\n"
                f"Picture ID: {element.get('picture_id')}\n"
                f"Page: {page}\n\n"
                f"{desc}"
            )
            add_element(element, visual_text)
            flush_chunk()
            continue

        # Normal text
        if el_type == "text":
            text = clean_text(element.get("text") or element.get("content") or "")
            if not text or is_low_value_fragment(text):
                continue
            sentences = split_sentences(text)
            for sentence in sentences:
                add_text_sentence(element, sentence)

    flush_chunk()

    # Merge very small chunks
    clean_chunks = []
    for chunk in chunks:
        content = clean_text(chunk["content"])
        if len(content) >= MIN_STANDALONE_CHARS:
            chunk["content"] = content
            clean_chunks.append(chunk)
            continue

        if clean_chunks:
            prev = clean_chunks[-1]
            merged = prev["content"] + "\n\n" + content
            if len(merged) <= MAX_CHARS:
                prev["content"] = merged
                prev["page_end"] = max(prev["page_end"] or 0, chunk["page_end"] or 0)
                prev["element_ids"].extend(chunk["element_ids"])
                prev["element_types"].extend(chunk["element_types"])
                prev["has_table"] = prev["has_table"] or chunk["has_table"]
                prev["has_visual"] = prev["has_visual"] or chunk["has_visual"]
                continue

        chunk["content"] = content
        clean_chunks.append(chunk)

    for idx, ch in enumerate(clean_chunks):
        ch["chunk_id"] = f"chunk_{idx:06d}"

    return clean_chunks


# ---------------------------------------------------------------------------
# 6. JINA v4 EMBEDDINGS & FAISS VECTOR STORE
# ---------------------------------------------------------------------------

def embed_chunks_jina(
    chunks: List[Dict[str, Any]],
    jina_api_key: str,
    progress_cb: Optional[Callable] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate normalized Jina v4 embeddings (2048-dim) in safe batches."""
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    texts = [
        f"title: {chunk.get('section') or 'none'} | text: {clean_text(chunk.get('content', ''))}"
        for chunk in chunks
    ]

    total_chunks = len(texts)
    embeddings_list = [None] * total_chunks

    for batch_start in range(0, total_chunks, JINA_EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + JINA_EMBEDDING_BATCH_SIZE, total_chunks)
        batch_texts = texts[batch_start:batch_end]

        if progress_cb:
            pct = 65 + int((batch_start / max(total_chunks, 1)) * 25)
            progress_cb(pct, f"Generating Jina v4 embeddings: chunks {batch_start+1}-{batch_end} of {total_chunks}...")

        payload = {
            "model": JINA_EMBEDDING_MODEL_NAME,
            "input": batch_texts,
            "task": "retrieval.passage",
            "embedding_type": "float",
            "dimensions": JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
            "truncate": False
        }

        success = False
        for attempt in range(1, JINA_EMBEDDING_MAX_RETRIES + 1):
            try:
                time.sleep(JINA_EMBEDDING_MIN_REQUEST_INTERVAL)
                resp = requests.post(
                    JINA_EMBEDDING_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Jina API {resp.status_code}: {resp.text[:200]}")

                data = resp.json().get("data", [])
                data_sorted = sorted(data, key=lambda it: int(it["index"]))
                for local_i, item in enumerate(data_sorted):
                    vec = np.asarray(item["embedding"], dtype="float32")
                    embeddings_list[batch_start + local_i] = vec
                success = True
                break
            except Exception as exc:
                if attempt == JINA_EMBEDDING_MAX_RETRIES:
                    raise RuntimeError(f"Failed embedding batch starting at {batch_start}: {exc}")
                time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 15))

        if not success:
            raise RuntimeError(f"Failed to obtain embeddings for batch {batch_start}-{batch_end}")

        time.sleep(JINA_EMBEDDING_INTER_BATCH_DELAY)

    embeddings_matrix = np.vstack(embeddings_list).astype("float32")

    # Defensively normalize to unit length for Inner Product = Cosine Similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix = embeddings_matrix / np.clip(norms, 1e-12, None)

    metadata = [
        {
            "faiss_index": i,
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section": chunk.get("section", ""),
            "has_table": chunk.get("has_table", False),
            "has_visual": chunk.get("has_visual", False)
        }
        for i, chunk in enumerate(chunks)
    ]

    return embeddings_matrix, metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS IndexFlatIP (Cosine Similarity on L2 normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# 7. QUERY RETRIEVAL & EXACT SEARCH
# ---------------------------------------------------------------------------

def embed_query_jina(query: str, jina_api_key: str) -> np.ndarray:
    """Embed search query with Jina task 'retrieval.query'."""
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JINA_EMBEDDING_MODEL_NAME,
        "input": [query],
        "task": "retrieval.query",
        "embedding_type": "float",
        "dimensions": JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
        "truncate": False
    }

    for attempt in range(1, JINA_EMBEDDING_MAX_RETRIES + 1):
        try:
            resp = requests.post(JINA_EMBEDDING_ENDPOINT, headers=headers, json=payload, timeout=45)
            if resp.status_code != 200:
                raise RuntimeError(f"Jina API {resp.status_code}: {resp.text[:200]}")
            data = resp.json().get("data", [])
            if not data:
                raise ValueError("Jina returned no embedding for query.")
            vec = np.asarray(data[0]["embedding"], dtype="float32")
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            return vec.reshape(1, -1)
        except Exception as exc:
            if attempt == JINA_EMBEDDING_MAX_RETRIES:
                raise
            time.sleep(1.0)


def retrieve_chunks(
    query: str,
    index_obj: DocuMindIndex,
    jina_api_key: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve top_k most relevant chunks using FAISS inner product."""
    if not index_obj or not index_obj.faiss_index or index_obj.faiss_index.ntotal == 0:
        return []

    q_vec = embed_query_jina(query, jina_api_key)
    scores, indices = index_obj.faiss_index.search(q_vec, min(top_k, index_obj.faiss_index.ntotal))

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk": index_obj.chunks[idx],
            "metadata": index_obj.metadata[idx]
        })
    return results


def search_exact_word(
    word_or_phrase: str,
    index_obj: DocuMindIndex
) -> List[Dict[str, Any]]:
    """
    Direct Keyword / Exact Match Search:
    Scans all chunks for exact word or phrase match, returning matching chunks with page numbers.
    """
    target = word_or_phrase.strip()
    if not target or not index_obj or not index_obj.chunks:
        return []

    matches = []
    # Build word boundary regex if it is a single word, or literal search if phrase
    pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE) if " " not in target else re.compile(re.escape(target), re.IGNORECASE)

    for chunk in index_obj.chunks:
        content = chunk.get("content", "")
        if pattern.search(content):
            # Extract matching snippet around first match
            match = pattern.search(content)
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(content), match.end() + 100)
            snippet = content[start_pos:end_pos].strip()
            if start_pos > 0:
                snippet = "..." + snippet
            if end_pos < len(content):
                snippet = snippet + "..."

            matches.append({
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk.get("section", ""),
                "has_table": chunk.get("has_table", False),
                "has_visual": chunk.get("has_visual", False),
                "snippet": snippet,
                "full_content": content
            })

    return matches


# ---------------------------------------------------------------------------
# 8. GROUNDED GEMINI ANSWER GENERATION
# ---------------------------------------------------------------------------

def build_retrieval_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved sources for prompt injection."""
    parts = []
    for item in results:
        chunk = item["chunk"]
        parts.append(
            f"""
SOURCE {item["rank"]}
Similarity score: {item["score"]:.4f}
Chunk ID: {chunk["chunk_id"]}
Page: {chunk.get("page_start")} - {chunk.get("page_end")}
Section: {chunk.get("section") or "General"}
Types: {", ".join(chunk.get("element_types", []))}

CONTENT:
{chunk["content"]}
""".strip()
        )
    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(parts)


def generate_answer(
    question: str,
    retrieved_results: List[Dict[str, Any]],
    gemini_api_key: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, str, List[int], List[str]]:
    """
    Generate grounded, factual answer using Google Gemini.
    Returns (answer_text, model_used, cited_pages, cited_items).
    """
    if not retrieved_results:
        return "I could not find any relevant information in the uploaded document for this question.", "none", [], []

    client = genai.Client(api_key=gemini_api_key)
    context_str = build_retrieval_context(retrieved_results)

    # Collect cited pages & figure/table indicators
    cited_pages = set()
    cited_items = []
    for r in retrieved_results:
        ch = r["chunk"]
        if ch.get("page_start") is not None:
            cited_pages.add(ch["page_start"])
        if ch.get("page_end") is not None:
            cited_pages.add(ch["page_end"])
        if ch.get("has_table"):
            cited_items.append(f"Table (Pg {ch.get('page_start')})")
        if ch.get("has_visual"):
            cited_items.append(f"Figure/Chart (Pg {ch.get('page_start')})")

    conversation_context = ""
    if chat_history and len(chat_history) > 1:
        history_snippets = []
        for msg in chat_history[-4:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_snippets.append(f"{role}: {msg['content']}")
        conversation_context = "\n\nRECENT CONVERSATION HISTORY:\n" + "\n".join(history_snippets)

    prompt = f"""
You are DocuMind, an intelligent document question-answering assistant.

Answer the user's question using ONLY the retrieved document context below.

IMPORTANT GROUNDING RULES:
1. Do not use outside knowledge or hallucinate facts.
2. Do not invent information. If the context does not contain enough information, clearly state: "The document does not provide this information."
3. Carefully compare all retrieved sources before answering. Prefer the source that directly addresses the specific query.
4. Pay close attention to dates, categories, units, metrics, geographic scope, and time periods.
5. Preserve numerical values and statistics exactly from the document.
6. If a calculation is required, perform it step-by-step using only numbers given in the context.
7. Mention the relevant page numbers, tables, or figures where the information is found.
8. Format your answer with clean Markdown (headings, bullet points, bold key terms). Never use raw LaTeX. Use % for percentages.
{conversation_context}

RETRIEVED DOCUMENT CONTEXT:
==========================
{context_str}
==========================

USER QUESTION:
=============
{question}
=============

FINAL ANSWER:
Provide an accurate, well-structured, and complete answer based strictly on the retrieved document context.
"""

    answer = None
    model_used = None

    for model_name in GEMINI_MODELS:
        for attempt in range(1, 4):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if resp.text and resp.text.strip():
                    answer = resp.text.strip()
                    model_used = model_name
                    break
            except Exception as exc:
                err = str(exc).upper()
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(2.0)
                    continue
                break
        if answer:
            break

    if not answer:
        answer = "I encountered an issue generating an answer. Please check your API quotas or rephrase your question."
        model_used = "error"

    return answer, model_used or "unknown", sorted(list(cited_pages)), list(set(cited_items))


# ---------------------------------------------------------------------------
# 9. END-TO-END PIPELINE PROCESSOR
# ---------------------------------------------------------------------------

def process_document(
    pdf_path: str,
    gemini_api_key: str,
    jina_api_key: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> DocuMindIndex:
    """
    Runs the complete 7-stage DocuMind pipeline:
      1. Setup & Ingestion
      2. Table & Picture Extraction
      3. Gemini Multimodal Vision Understanding
      4. Canonical Assembly
      5. Structure-Aware Chunking
      6. Jina v4 Embeddings & FAISS Index
      7. Return ready-to-query DocuMindIndex object
    """
    start_time = time.time()
    work_dir = Path(pdf_path).parent / f"documind_{Path(pdf_path).stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    def report(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # Stage 1 & 2: Ingestion & Document Parsing
    report(5, "Stage 1/6: Opening PDF and parsing document elements...")
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    docling_data, raw_pictures = parse_pdf(pdf_path, work_dir, report)

    # Filter visual candidates
    visual_candidates = []
    for pic in raw_pictures:
        w = pic.get("bbox_width", pic.get("width", 0))
        h = pic.get("bbox_height", pic.get("height", 0))
        if w >= VISUAL_MIN_WIDTH and h >= VISUAL_MIN_HEIGHT:
            visual_candidates.append(pic)

    # Stage 3: Gemini Multimodal Vision Understanding
    report(35, f"Stage 2/6: Analyzing {len(visual_candidates)} visual elements with Google Gemini...")
    gemini_visuals = describe_visuals_with_gemini(visual_candidates, gemini_api_key, report)

    # Stage 4: Canonical Assembly
    report(60, "Stage 3/6: Assembling canonical document elements...")
    canonical_elements = build_canonical_document(docling_data, gemini_visuals, raw_pictures)

    # Stage 5: Structure-Aware Chunking
    report(65, "Stage 4/6: Generating structure-aware chunks (preserving tables & visuals)...")
    chunks = create_structure_aware_chunks(canonical_elements)

    if not chunks:
        raise ValueError("No text or content chunks could be extracted from the document.")

    # Stage 6: Jina v4 Embeddings & FAISS
    report(75, f"Stage 5/6: Embedding {len(chunks)} chunks with Jina v4 (2048 dimensions)...")
    embeddings, metadata = embed_chunks_jina(chunks, jina_api_key, report)

    report(95, "Stage 6/6: Building FAISS inner-product vector index...")
    faiss_index = build_faiss_index(embeddings)

    elapsed = round(time.time() - start_time, 2)
    report(100, f"Processing complete in {elapsed} seconds!")

    # Calculate statistics
    total_tables = sum(1 for c in chunks if c.get("has_table"))
    successful_visuals = sum(1 for v in gemini_visuals.values() if v.get("status") == "success")
    total_visual_chunks = sum(1 for c in chunks if c.get("has_visual"))
    total_visuals = max(successful_visuals, total_visual_chunks, len(visual_candidates))

    stats = DocumentStats(
        total_pages=total_pages,
        total_chunks=len(chunks),
        total_tables=total_tables,
        total_pictures=total_visuals,
        total_elements=len(canonical_elements),
        embedding_dim=JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
        processing_time_sec=elapsed
    )

    return DocuMindIndex(
        chunks=chunks,
        metadata=metadata,
        faiss_index=faiss_index,
        stats=stats,
        visual_manifest=visual_candidates,
        canonical_elements=canonical_elements,
        pdf_path=pdf_path,
        pdf_name=Path(pdf_path).name
    )
