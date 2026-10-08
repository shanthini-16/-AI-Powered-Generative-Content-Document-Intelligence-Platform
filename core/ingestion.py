"""
Document Ingestion and Preprocessing Module
Supports PDF, DOCX, TXT, MD, CSV, JSON, and HTML.
Provides text normalization, metadata profiling, section extraction, and advanced chunking strategies.
"""

import os
import re
import csv
import json
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    token_estimate: int
    word_count: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestedDocument:
    doc_id: str
    filename: str
    file_type: str
    raw_text: str
    clean_text: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    sections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "clean_text_preview": self.clean_text[:400] + ("..." if len(self.clean_text) > 400 else ""),
            "chunks_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
            "statistics": self.statistics,
            "sections": self.sections,
        }


class DocumentIngestor:
    """Ingests multi-format documents, cleans, profiles, and chunks them."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}

    def __init__(self, default_chunk_size: int = 500, default_chunk_overlap: int = 100):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def ingest_file(
        self,
        file_path: str,
        chunk_strategy: str = "paragraph",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> IngestedDocument:
        """Ingests a file from disk with format-specific parsing."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}")

        file_bytes = path.read_bytes()
        file_size_bytes = len(file_bytes)
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        filename = path.name

        raw_text, doc_meta = self._parse_format(file_bytes, ext, filename)

        doc_meta.update({
            "source_path": str(path.resolve()),
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_bytes / 1024, 2),
            "content_hash": file_hash,
            "extension": ext,
        })

        return self.process_raw_text(
            raw_text=raw_text,
            filename=filename,
            file_type=ext.lstrip("."),
            metadata=doc_meta,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
        )

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_strategy: str = "paragraph",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> IngestedDocument:
        """Ingests a file uploaded as bytes or in-memory buffer."""
        ext = Path(filename).suffix.lower() or ".txt"
        file_size_bytes = len(file_bytes)
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

        raw_text, doc_meta = self._parse_format(file_bytes, ext, filename)
        doc_meta.update({
            "source": "upload_buffer",
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_bytes / 1024, 2),
            "content_hash": file_hash,
            "extension": ext,
        })

        return self.process_raw_text(
            raw_text=raw_text,
            filename=filename,
            file_type=ext.lstrip("."),
            metadata=doc_meta,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
        )

    def ingest_direct_text(
        self,
        text: str,
        filename: str = "direct_input.txt",
        chunk_strategy: str = "paragraph",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> IngestedDocument:
        """Ingests raw text directly from UI or API."""
        return self.process_raw_text(
            raw_text=text,
            filename=filename,
            file_type="txt",
            metadata={"source": "direct_text_input"},
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
        )

    def _parse_format(self, file_bytes: bytes, ext: str, filename: str) -> tuple[str, Dict[str, Any]]:
        """Parses raw bytes into string based on extension."""
        meta: Dict[str, Any] = {}

        if ext == ".pdf":
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                num_pages = len(reader.pages)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    page_t = page.extract_text() or ""
                    if page_t.strip():
                        pages_text.append(f"--- [Page {idx+1}] ---\n{page_t}")
                raw_text = "\n\n".join(pages_text)
                meta["page_count"] = num_pages
                if reader.metadata:
                    meta["pdf_author"] = reader.metadata.author or ""
                    meta["pdf_title"] = reader.metadata.title or ""
                    meta["pdf_creator"] = reader.metadata.creator or ""
            except Exception as e:
                raw_text = f"Error reading PDF: {e}"

        elif ext == ".docx":
            try:
                import docx
                import io
                doc = docx.Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                # Also include tables
                table_texts = []
                for table in doc.tables:
                    rows = []
                    for row in table.rows:
                        row_vals = [cell.text.strip() for cell in row.cells]
                        rows.append(" | ".join(row_vals))
                    if rows:
                        table_texts.append("\n".join(rows))
                raw_text = "\n\n".join(paragraphs + table_texts)
                meta["paragraph_count"] = len(paragraphs)
                meta["table_count"] = len(doc.tables)
            except Exception as e:
                raw_text = f"Error reading DOCX: {e}"

        elif ext == ".csv":
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
                reader = csv.reader(text_content.splitlines())
                rows = list(reader)
                if rows:
                    header = rows[0]
                    formatted_rows = [f"Columns: {', '.join(header)}"]
                    for idx, row in enumerate(rows[1:], start=1):
                        formatted_rows.append(f"Record {idx}: " + ", ".join(f"{h}: {v}" for h, v in zip(header, row)))
                    raw_text = "\n".join(formatted_rows)
                    meta["row_count"] = len(rows)
                    meta["column_count"] = len(header)
                else:
                    raw_text = text_content
            except Exception as e:
                raw_text = f"Error reading CSV: {e}"

        elif ext == ".json":
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
                parsed_json = json.loads(text_content)
                raw_text = json.dumps(parsed_json, indent=2)
                meta["is_json_valid"] = True
                meta["json_keys"] = list(parsed_json.keys()) if isinstance(parsed_json, dict) else len(parsed_json)
            except Exception as e:
                raw_text = file_bytes.decode("utf-8", errors="replace")
                meta["is_json_valid"] = False

        elif ext in {".html", ".htm"}:
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
                # Strip HTML tags cleanly
                clean = re.sub(r"<style.*?</style>", "", text_content, flags=re.DOTALL)
                clean = re.sub(r"<script.*?</script>", "", clean, flags=re.DOTALL)
                clean = re.sub(r"<[^>]+>", " ", clean)
                raw_text = re.sub(r"\s+", " ", clean).strip()
            except Exception as e:
                raw_text = file_bytes.decode("utf-8", errors="replace")

        else:
            # Plain text, markdown, etc.
            raw_text = file_bytes.decode("utf-8", errors="replace")

        return raw_text, meta

    def preprocess_text(self, text: str) -> str:
        """Cleans, normalizes unicode, removes artifacts, standardizes whitespace."""
        if not text:
            return ""

        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # Replace smart quotes and dashes
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--", "\u2026": "...", "\r\n": "\n", "\r": "\n"
        }
        for k, v in replacements.items():
            normalized = normalized.replace(k, v)

        # Remove control characters except standard whitespace
        normalized = "".join(ch for ch in normalized if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

        # Normalize multiple spaces per line
        lines = []
        for line in normalized.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)

        # Collapse excessive newlines (max 2 consecutive)
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return cleaned.strip()

    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identifies headers, sections, or numbered chapters."""
        sections: List[Dict[str, Any]] = []
        pattern = re.compile(
            r"^(#{1,6}\s+.+|[A-Z0-9\.\s]{3,50}:|[0-9]+\.[0-9]*\s+[A-Z][a-zA-Z\s]+|SECTION\s+[0-9A-Z]+.+|ARTICLE\s+[0-9A-Z]+.+)$",
            re.MULTILINE
        )

        matches = list(pattern.finditer(text))
        if not matches:
            return [{"title": "General", "start": 0, "end": len(text), "content_preview": text[:200]}]

        for i, match in enumerate(matches):
            title = match.group(0).strip("# ").strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append({
                "title": title,
                "start": start,
                "end": end,
                "length": len(content),
                "content_preview": content[:250] + ("..." if len(content) > 250 else "")
            })

        return sections

    def calculate_statistics(self, text: str) -> Dict[str, Any]:
        """Calculates linguistic and statistical metrics for the text."""
        words = re.findall(r"\b[A-Za-z0-9_\-']+\b", text)
        sentences = re.split(r"[.!?]+(?:\s+|$)", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_len = round(word_count / max(sentence_count, 1), 1)
        avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)

        # Lexical diversity (TTR)
        unique_words = set(w.lower() for w in words)
        lexical_diversity = round(len(unique_words) / max(word_count, 1), 3)

        # Estimated tokens (GPT/Gemini rule of thumb: ~4 chars or ~0.75 words per token)
        estimated_tokens = int(char_count / 3.8) if char_count > 0 else 0

        # Frequent keywords (simple stopword filter)
        stopwords = {
            "the", "and", "to", "of", "a", "in", "that", "is", "for", "on", "it",
            "as", "with", "was", "at", "by", "an", "be", "this", "which", "or",
            "from", "but", "not", "are", "we", "has", "will", "have", "you", "they"
        }
        word_freq: Dict[str, int] = {}
        for w in words:
            wl = w.lower()
            if len(wl) > 3 and wl not in stopwords and not wl.isnumeric():
                word_freq[wl] = word_freq.get(wl, 0) + 1

        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "estimated_tokens": estimated_tokens,
            "avg_sentence_length": avg_sentence_len,
            "avg_word_length": avg_word_len,
            "lexical_diversity": lexical_diversity,
            "top_keywords": [{"word": k, "frequency": v} for k, v in top_keywords]
        }

    def chunk_text(
        self,
        text: str,
        strategy: str = "paragraph",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[DocumentChunk]:
        """
        Splits text into chunks using requested strategy:
        - 'paragraph': splits on double newlines while respecting chunk size bounds.
        - 'fixed_window': sliding word-based window with specified overlap.
        - 'section': splits according to identified headers/sections.
        """
        if not text:
            return []

        chunks: List[DocumentChunk] = []

        if strategy == "fixed_window":
            words = text.split()
            step = max(1, chunk_size - chunk_overlap)
            chunk_id = 1
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                chunk_str = " ".join(chunk_words)
                token_est = int(len(chunk_str) / 3.8)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    token_estimate=token_est,
                    word_count=len(chunk_words),
                    start_char=text.find(chunk_words[0]) if chunk_words else 0,
                    end_char=text.find(chunk_words[-1]) + len(chunk_words[-1]) if chunk_words else 0,
                    metadata={"strategy": "fixed_window", "window_start": i, "window_end": i + len(chunk_words)}
                ))
                chunk_id += 1
                if i + chunk_size >= len(words):
                    break

        elif strategy == "section":
            sections = self.extract_sections(text)
            for idx, sec in enumerate(sections, start=1):
                sec_text = text[sec["start"]:sec["end"]].strip()
                if not sec_text:
                    continue
                w_count = len(sec_text.split())
                chunks.append(DocumentChunk(
                    chunk_id=idx,
                    text=sec_text,
                    token_estimate=int(len(sec_text) / 3.8),
                    word_count=w_count,
                    start_char=sec["start"],
                    end_char=sec["end"],
                    metadata={"strategy": "section", "section_title": sec["title"]}
                ))

        else:
            # Paragraph-based (default)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_chunk_paragraphs: List[str] = []
            current_words = 0
            chunk_id = 1
            running_char = 0

            for p in paragraphs:
                p_words = len(p.split())
                if current_words + p_words > chunk_size and current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs)
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_str,
                        token_estimate=int(len(chunk_str) / 3.8),
                        word_count=len(chunk_str.split()),
                        start_char=running_char,
                        end_char=running_char + len(chunk_str),
                        metadata={"strategy": "paragraph", "num_paragraphs": len(current_chunk_paragraphs)}
                    ))
                    chunk_id += 1
                    running_char += len(chunk_str) + 2

                    # Apply overlap: retain last paragraph if small enough
                    if p_words <= chunk_overlap:
                        current_chunk_paragraphs = [current_chunk_paragraphs[-1], p]
                        current_words = len(current_chunk_paragraphs[0].split()) + p_words
                    else:
                        current_chunk_paragraphs = [p]
                        current_words = p_words
                else:
                    current_chunk_paragraphs.append(p)
                    current_words += p_words

            if current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    token_estimate=int(len(chunk_str) / 3.8),
                    word_count=len(chunk_str.split()),
                    start_char=running_char,
                    end_char=running_char + len(chunk_str),
                    metadata={"strategy": "paragraph", "num_paragraphs": len(current_chunk_paragraphs)}
                ))

        return chunks

    def process_raw_text(
        self,
        raw_text: str,
        filename: str,
        file_type: str,
        metadata: Dict[str, Any],
        chunk_strategy: str = "paragraph",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> IngestedDocument:
        """Orchestrates cleaning, profiling, chunking, and returns IngestedDocument."""
        clean = self.preprocess_text(raw_text)
        stats = self.calculate_statistics(clean)
        sections = self.extract_sections(clean)
        chunks = self.chunk_text(clean, strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_id = hashlib.md5(f"{filename}-{clean[:100]}".encode()).hexdigest()[:12]

        return IngestedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            raw_text=raw_text,
            clean_text=clean,
            chunks=chunks,
            metadata=metadata,
            statistics=stats,
            sections=sections,
        )
