"""
Project-IQ: Privacy Sanitization & Front-Matter Cleaning Pipeline
Removes administrative front-matter (certificates, declarations, plagiarism summaries)
and redacts sensitive PII (student IDs, PRNs, emails, phones, local names) while
preserving public scholarly citations and literature references.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import spacy

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.cleaning")

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+\s*(?:@|\[at\]|\(at\))\s*[A-Z0-9.-]+\s*(?:\.|\[dot\]|\(dot\))\s*[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
CREDENTIAL_RE = re.compile(r"(?i)\b(password|passwd|pwd|api[_ -]?key|secret|access[_ -]?token|token)\s*[:=]\s*\S+")
STUDENT_ID_RE = re.compile(r"(?i)\b(?:(?:student\s*id|prn|roll(?:\s*no)?|enrollment(?:\s*(?:no|number))?|registration(?:\s*(?:no|number))?)\s*[:#=\-]?\s*)+\[?[A-Z0-9][A-Z0-9/_-]{2,}\]?")
ROLE_RE = re.compile(r"(?i)\b(?:submitted\s+by|guided\s+by|guide|mentor|supervisor|faculty|student(?:\s+name)?|team\s+member|coordinator|prepared\s+by|author(?:s)?\s+of\s+project)\b")

ADMIN_PHRASES = [
    "this is to certify", "bonafide certificate", "certificate of completion",
    "acknowledgement", "acknowledgments", "declaration of candidate",
    "candidate's declaration", "plagiarism report", "similarity report",
    "turnitin", "sponsorship letter", "submission summary",
    "conference management toolkit"
]

HEADING_EXCLUSIONS = {
    "CHAPTER", "ABSTRACT", "INTRODUCTION", "LITERATURE SURVEY", "LITERATURE REVIEW",
    "METHODOLOGY", "PROPOSED SYSTEM", "SYSTEM ARCHITECTURE", "IMPLEMENTATION",
    "RESULTS", "RESULT AND DISCUSSION", "CONCLUSION", "FUTURE SCOPE", "REFERENCES",
    "EXPECTED OUTCOME", "PROBLEM STATEMENT", "OBJECTIVES"
}

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.debug("spaCy 'en_core_web_sm' model not found, falling back to blank english pipeline.")
            _nlp = spacy.blank("en")
    return _nlp


def clean_text(s: str | None) -> str:
    """Clean, normalize and sanitize raw text strings, stripping control characters."""
    if not s:
        return ""
    s = s.replace("\x00", " ").replace("\u00ad", " ")
    s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    s = s.replace("[at]", "@").replace("(at)", "@").replace("[dot]", ".").replace("(dot)", ".")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


norm = clean_text


def is_admin_front_matter(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in ADMIN_PHRASES)


def heading_like(text: str) -> bool:
    t = norm(text)
    if not t or len(t) > 180:
        return False
    if re.match(r"(?i)^(?:chapter\s+)?(?:[ivxlc]+|\d+(?:\.\d+)*)[.)]?\s+", t):
        return True
    if t.isupper() and 2 <= len(t.split()) <= 18:
        return True
    return len(t.split()) <= 9 and not t.endswith((".", "?", "!", ";", ":"))


def discover_local_names(blocks: List[Dict[str, Any]], nlp) -> List[str]:
    names = set()
    total_blocks = len(blocks)

    for idx, b in enumerate(blocks):
        text = norm(b.get("text", ""))
        if not text:
            continue

        # Scan if in initial administrative zone (first 25 blocks) or contains role markers
        is_front_zone = idx < min(25, total_blocks)
        has_role = bool(ROLE_RE.search(text))

        if not (is_front_zone or has_role):
            continue

        # Run NER on the text
        doc = nlp(text[:8000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = norm(ent.text)
                # Filter out single words or non-name entities
                if 2 <= len(name.split()) <= 4 and len(name) >= 5:
                    if name.upper() not in HEADING_EXCLUSIONS:
                        names.add(name)

        # Also inspect line-by-line for explicit role matches and ALL-CAPS names
        for line in text.splitlines():
            line_str = norm(line)
            if ROLE_RE.search(line_str):
                line_doc = nlp(line_str)
                for ent in line_doc.ents:
                    if ent.label_ == "PERSON":
                        name = norm(ent.text)
                        if 2 <= len(name.split()) <= 4:
                            names.add(name)
            elif is_front_zone and line_str.isupper() and 2 <= len(line_str.split()) <= 4:
                # Bare uppercase student name in front-matter zone
                if line_str not in HEADING_EXCLUSIONS and not any(h in line_str for h in HEADING_EXCLUSIONS):
                    names.add(line_str.title())
                    names.add(line_str)

    return sorted(names, key=lambda x: (-len(x), x.lower()))


def redact_text(text: str, local_names: List[str]) -> Tuple[str, Dict[str, int]]:
    text = norm(text)
    counts = {"email": 0, "phone": 0, "credential": 0, "student_id": 0, "local_name": 0}

    def _sub_email(m):
        counts["email"] += 1
        return "[EMAIL]"

    def _sub_phone(m):
        counts["phone"] += 1
        return "[PHONE]"

    def _sub_cred(m):
        counts["credential"] += 1
        return f"{m.group(1)} = [REDACTED_CREDENTIAL]"

    def _sub_sid(m):
        counts["student_id"] += 1
        return "[STUDENT_ID]"

    text = EMAIL_RE.sub(_sub_email, text)
    text = PHONE_RE.sub(_sub_phone, text)
    text = CREDENTIAL_RE.sub(_sub_cred, text)
    text = STUDENT_ID_RE.sub(_sub_sid, text)

    # Redact discovered local names
    for name in local_names:
        escaped = re.escape(name)
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.I)
        matches = len(pattern.findall(text))
        if matches > 0:
            counts["local_name"] += matches
            text = pattern.sub("[NAME]", text)

    # Clean redundant consecutive [NAME] tags
    text = re.sub(r"(\[NAME\](\s*\[NAME\])+)", "[NAME]", text)
    return text, counts


def stable_id(*parts: Any) -> str:
    """Generate deterministic stable block ID using SHA-1 hex digest of serialized parts."""
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def clean_record_data(data: Dict[str, Any], nlp) -> Dict[str, Any]:
    meta = data.get("document", {})
    raw_blocks = data.get("blocks", [])

    # Legacy flat format fallback
    if not raw_blocks and "text" in data:
        raw_blocks = [{
            "block_id": stable_id(meta.get("document_id", "0"), 0, "paragraph"),
            "document_id": meta.get("document_id", "0"),
            "block_type": "paragraph",
            "text": data["text"],
            "ocr": False
        }]

    local_names = discover_local_names(raw_blocks, nlp)
    cleaned_blocks = []
    total_stats = {"email": 0, "phone": 0, "credential": 0, "student_id": 0, "local_name": 0}

    for b in raw_blocks:
        raw_text = b.get("text", "")
        if not raw_text or is_admin_front_matter(raw_text):
            continue

        sanitized_text, stats = redact_text(raw_text, local_names)
        if not sanitized_text.strip():
            continue

        for k, v in stats.items():
            total_stats[k] += v

        cb = dict(b)
        cb["text"] = sanitized_text
        cb["is_heading"] = heading_like(sanitized_text)
        cb.pop("source_path", None)
        cleaned_blocks.append(cb)

    combined_text = "\n\n".join(b["text"] for b in cleaned_blocks)

    return {
        "document": {k: v for k, v in meta.items() if k != "source_path"},
        "privacy": {
            "automatic_local_names_found": len(local_names),
            "redaction_counts": total_stats,
            "policy": "Local student and guide identities detected and redacted; academic citations preserved."
        },
        "block_count": len(cleaned_blocks),
        "text": combined_text,
        "blocks": cleaned_blocks
    }


def clean_all(input_dir: str | Path, output_dir: str | Path, report_dir: str | Path | None = None) -> Dict[str, Any]:
    inp = Path(input_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)

    nlp = get_nlp()
    seen_hashes: Set[str] = set()
    cleaned_count = 0
    total_redactions = {"email": 0, "phone": 0, "credential": 0, "student_id": 0, "local_name": 0}

    for json_file in sorted(inp.glob("*.json")):
        if json_file.name == "_manifest.json":
            continue

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            meta = data.get("document", {})
            sha = meta.get("file_sha256")

            # Deduplication check
            if sha and sha in seen_hashes:
                continue
            if sha:
                seen_hashes.add(sha)

            cleaned_doc = clean_record_data(data, nlp)
            out_file = out / f"{meta.get('document_id', json_file.stem)}_cleaned.json"
            out_file.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
            cleaned_count += 1

            for k, v in cleaned_doc["privacy"]["redaction_counts"].items():
                total_redactions[k] += v

            logger.info(f"CLEANED: {meta.get('source_file', json_file.name)} -> {out_file.name}")
        except Exception as e:
            logger.error(f"ERROR cleaning {json_file.name}: {repr(e)}")

    summary = {
        "cleaned_documents": cleaned_count,
        "total_redactions": total_redactions
    }

    if report_dir:
        report_file = Path(report_dir) / "cleaning_report.json"
        report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Sanitize PII and front-matter from extracted documents.")
    parser.add_argument("--input", required=True, help="Input directory of extracted JSON files")
    parser.add_argument("--output", required=True, help="Output directory for cleaned JSON files")
    parser.add_argument("--report", default=None, help="Optional directory for cleaning_report.json")
    args = parser.parse_args()

    summary = clean_all(args.input, args.output, args.report)
    logger.info(f"CLEANING COMPLETE: {summary['cleaned_documents']} documents sanitized.")
    logger.info(f"Redactions: {summary['total_redactions']}")


if __name__ == "__main__":
    main()
