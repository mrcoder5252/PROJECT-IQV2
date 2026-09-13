"""
Project-IQ: Structure-Aware, Token-Bounded Chunking & Deduplication
Splits cleaned documents into section-aware chunks that never cross headings,
strictly bounds token size (target ~450 tokens, hard limit 520 tokens),
detects and clusters duplicate or near-duplicate passages across documents,
and attaches ML-ready structural features.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import tiktoken

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.chunking")

CHUNK_SIZE_TOKENS = config.target_chunk_tokens
CHUNK_OVERLAP_TOKENS = config.chunk_overlap_tokens
HARD_MAX_TOKENS = config.hard_max_tokens
MIN_CHUNK_WORDS = config.min_chunk_words
SHINGLE_SIZE = config.shingle_size
MIN_SHARED_SHINGLES = config.min_shared_shingles
JACCARD_THRESHOLD = config.jaccard_threshold

_enc = None


def get_encoder():
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = tiktoken.encoding_for_model("gpt-4")
    return _enc


def count_tokens(text: str) -> int:
    enc = get_encoder()
    return len(enc.encode(text))


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\d*\s+)?(ABSTRACT|INTRODUCTION|LITERATURE (?:SURVEY|REVIEW)|"
    r"METHODOLOGY|PROPOSED SYSTEM|SYSTEM ARCHITECTURE|IMPLEMENTATION|RESULTS?|"
    r"RESULT AND DISCUSSION|CONCLUSION|FUTURE SCOPE|REFERENCES|CHAPTER\s*\d+|"
    r"PROBLEM STATEMENT|OBJECTIVES|SCOPE OF WORK)\b", re.MULTILINE | re.I
)


def strip_boilerplate(text: str) -> str:
    text = re.sub(r"\[NAME\](\s*\[NAME\])+", "[NAME]", text)
    text = re.sub(r"(?i)WAGHOLI,?\s*PUNE\s*\d{6}\s*\d{4}-\d{2}", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def split_by_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("BODY", text)]
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_name = m.group(1).upper()
        # Normalize section naming
        if "LITERATURE" in sec_name:
            sec_name = "LITERATURE SURVEY"
        elif "RESULT" in sec_name:
            sec_name = "RESULTS"
        sections.append((sec_name, text[start:end]))
    return sections


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_long_paragraph(para: str, max_tokens: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        tc = count_tokens(sent)
        if tc > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words = sent.split()
            piece_words: List[str] = []
            for w in words:
                piece_words.append(w)
                if count_tokens(" ".join(piece_words)) >= max_tokens:
                    chunks.append(" ".join(piece_words))
                    piece_words = []
            if piece_words:
                chunks.append(" ".join(piece_words))
            continue

        if current and current_len + tc > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += tc

    if current:
        chunks.append(" ".join(current))
    return chunks


def _enforce_hard_max(text: str, hard_max: int = HARD_MAX_TOKENS) -> str:
    """Guarantee text does not exceed hard token limit."""
    if count_tokens(text) <= hard_max:
        return text
    enc = get_encoder()
    tokens = enc.encode(text)
    if len(tokens) > hard_max:
        return enc.decode(tokens[:hard_max]).strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    paragraphs = split_into_paragraphs(text)
    units: List[str] = []
    for p in paragraphs:
        if count_tokens(p) > chunk_size:
            units.extend(split_long_paragraph(p, chunk_size))
        else:
            units.append(p)

    chunks: List[str] = []
    current_words: List[str] = []
    current_text_parts: List[str] = []

    def flush():
        if current_text_parts:
            assembled = " ".join(current_text_parts).strip()
            # Enforce hard maximum
            if count_tokens(assembled) > HARD_MAX_TOKENS:
                words = assembled.split()
                while words and count_tokens(" ".join(words)) > HARD_MAX_TOKENS:
                    words.pop()
                assembled = " ".join(words).strip()
            # Token level fallback guarantee
            assembled = _enforce_hard_max(assembled, HARD_MAX_TOKENS)
            if assembled:
                chunks.append(assembled)

    for unit in units:
        unit_tc = count_tokens(unit)
        if unit_tc > chunk_size:
            if current_words:
                flush()
                current_words.clear()
                current_text_parts.clear()
            words = unit.split()
            piece_words: List[str] = []
            for w in words:
                piece_words.append(w)
                if count_tokens(" ".join(piece_words)) >= chunk_size:
                    chunk_str = _enforce_hard_max(" ".join(piece_words), HARD_MAX_TOKENS)
                    if chunk_str:
                        chunks.append(chunk_str)
                    piece_words = []
            if piece_words:
                chunk_str = _enforce_hard_max(" ".join(piece_words), HARD_MAX_TOKENS)
                if chunk_str:
                    chunks.append(chunk_str)
            continue

        current_tc = count_tokens(" ".join(current_words)) if current_words else 0
        if current_words and current_tc + unit_tc > chunk_size:
            flush()
            tail_words = current_words[:]
            while tail_words and count_tokens(" ".join(tail_words)) > overlap:
                tail_words.pop(0)
            current_words = list(tail_words)
            current_text_parts = [" ".join(tail_words)] if tail_words else []

        current_words.extend(unit.split())
        current_text_parts.append(unit)

    flush()
    return chunks


def chunk_text_by_section(text: str) -> List[Tuple[str, str]]:
    result = []
    for section_name, section_text in split_by_sections(text):
        for piece in chunk_text(section_text):
            result.append((piece, section_name))
    return result


def extract_ml_features(text: str) -> Dict[str, Any]:
    return {
        "sentence_count": len(re.findall(r"[.!?]+[\"')\]]*(?:\s+|$)", text)),
        "has_numbers": bool(re.search(r"\d", text)),
        "has_citation": bool(re.search(r"\[\d+\]|\(\d{4}\)", text)),
    }


def shingles(text: str, n: int = SHINGLE_SIZE) -> Set[str]:
    words = re.findall(r"[\w\[\]]+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class UnionFind:
    def __init__(self, ids: List[str]):
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_duplicate_clusters(chunk_records: List[Dict[str, Any]]) -> Dict[str, str]:
    shingle_sets = {c["chunk_id"]: shingles(c["text"]) for c in chunk_records}
    inverted = defaultdict(list)
    for cid, sset in shingle_sets.items():
        for s in sset:
            inverted[s].append(cid)

    uf = UnionFind([c["chunk_id"] for c in chunk_records])
    for cid, sset in shingle_sets.items():
        if not sset:
            continue
        candidate_counts = defaultdict(int)
        for s in sset:
            for other_cid in inverted[s]:
                if other_cid != cid:
                    candidate_counts[other_cid] += 1

        for other_cid, shared in candidate_counts.items():
            if shared < MIN_SHARED_SHINGLES:
                continue
            other_sset = shingle_sets[other_cid]
            union_size = len(sset | other_sset)
            if union_size == 0:
                continue
            if len(sset & other_sset) / union_size >= JACCARD_THRESHOLD:
                uf.union(cid, other_cid)

    return {cid: uf.find(cid) for cid in shingle_sets}


def make_chunk_id(document_id: str, category: str, record_index: int, chunk_index: int) -> str:
    raw = f"{document_id}|{category}|{record_index}|{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def chunk_all(cleaned_dir: str | Path, output_file: str | Path) -> List[Dict[str, Any]]:
    inp = Path(cleaned_dir).resolve()
    out = Path(output_file).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    skipped_small = 0

    json_files = sorted(inp.glob("*.json"))
    for record_index, fpath in enumerate(json_files):
        if fpath.name == "cleaning_report.json":
            continue

        try:
            doc_data = json.loads(fpath.read_text(encoding="utf-8"))
            meta = doc_data.get("document", {})
            raw_text = doc_data.get("text", "")
            cleaned_text = strip_boilerplate(raw_text)
            if not cleaned_text.strip():
                continue

            doc_id = meta.get("document_id", fpath.stem)
            category = meta.get("category", "project_report")
            department = meta.get("department", "General")
            group_id = meta.get("group_id", "General")
            project_name = meta.get("project_name", "Unknown")
            source_file = meta.get("source_file", fpath.name)
            doc_type = meta.get("document_type", "pdf")
            doc_hash = content_hash(cleaned_text)

            pieces = chunk_text_by_section(cleaned_text)
            kept_index = 0

            for piece, section in pieces:
                words = piece.split()
                if len(words) < MIN_CHUNK_WORDS:
                    skipped_small += 1
                    continue

                chunk_id = make_chunk_id(doc_id, category, record_index, kept_index)
                tc = count_tokens(piece)
                ml_feats = extract_ml_features(piece)

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "source_file": source_file,
                    "department": department,
                    "group_id": group_id,
                    "project_name": project_name,
                    "document_type": doc_type,
                    "category": category,
                    "record_index": record_index,
                    "chunk_index": kept_index,
                    "section": section,
                    "word_count": len(words),
                    "token_count": tc,
                    "source_hash": doc_hash,
                    "text": piece,
                    **ml_feats
                })
                kept_index += 1

            for c in all_chunks[-kept_index:] if kept_index else []:
                c["total_chunks_in_record"] = kept_index

        except Exception as e:
            logger.error(f"Error chunking {fpath.name}: {repr(e)}")

    logger.info(f"Generated {len(all_chunks)} raw chunks (filtered out {skipped_small} fragments < {MIN_CHUNK_WORDS} words)")

    # Deduplication and clustering
    if all_chunks:
        cluster_map = find_duplicate_clusters(all_chunks)
        category_priority = {"research_paper": 0, "blackbook": 1, "review4": 2, "abstract_synopsis": 3, "project_report": 4}
        clusters = defaultdict(list)
        for c in all_chunks:
            clusters[cluster_map[c["chunk_id"]]].append(c)

        canonical_of_cluster = {}
        for cluster_id, members in clusters.items():
            if len(members) == 1:
                canonical_of_cluster[cluster_id] = members[0]["chunk_id"]
                continue
            best = min(members, key=lambda c: (category_priority.get(c["category"], 9), -c["word_count"]))
            canonical_of_cluster[cluster_id] = best["chunk_id"]

        for c in all_chunks:
            cid = c["chunk_id"]
            cluster_id = cluster_map[cid]
            canonical_id = canonical_of_cluster[cluster_id]
            c["duplicate_group_id"] = cluster_id
            c["is_canonical"] = (cid == canonical_id)
            c["duplicate_of_chunk_id"] = None if cid == canonical_id else canonical_id

    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    dupes = sum(1 for c in all_chunks if not c["is_canonical"])
    logger.info(f"SAVED {len(all_chunks)} chunks to {out}")
    logger.info(f"Canonical chunks: {len(all_chunks) - dupes} | Duplicate tagged chunks: {dupes}")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk cleaned documents for RAG and ML ingestion.")
    parser.add_argument("--input", required=True, help="Directory containing cleaned JSON files")
    parser.add_argument("--output", required=True, help="Output chunks.jsonl file path")
    args = parser.parse_args()

    chunk_all(args.input, args.output)


if __name__ == "__main__":
    main()
