"""
Project-IQ: Privacy Sanitization & Front-Matter Cleaning Pipeline
Removes administrative front-matter (certificates, declarations, plagiarism summaries)
and redacts sensitive PII (student IDs, PRNs, emails, phones, local names) while
preserving public scholarly citations and literature references.

Setup Instruction:
Ensure the spaCy English NER pipeline is installed before running:
    python -m spacy download en_core_web_sm
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

# Words that can never be part of a person's name; stops titles/headings being learned as "names".
NAME_STOPWORDS = {
    "system", "management", "project", "application", "using", "based", "detection", "analysis", "report",
    "department", "college", "engineering", "technology", "university", "institute", "learning", "prediction",
    "monitoring", "recognition", "classification", "intelligent", "smart", "automated", "automatic", "web",
    "mobile", "network", "data", "model", "design", "development", "review", "presentation", "synopsis",
    "abstract", "chapter", "semester", "academic", "year", "computer", "science", "information", "artificial",
    "intelligence", "machine", "deep", "iot", "blockchain", "platform", "framework", "approach", "study",
}


def looks_like_name(candidate: str, title: str = "") -> bool:
    words = candidate.replace(".", " ").split()
    if not 2 <= len(words) <= 4:
        return False
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in words):
        return False
    if any(w.lower() in NAME_STOPWORDS for w in words):
        return False
    if candidate.upper() in HEADING_EXCLUSIONS:
        return False
    if title and candidate.lower() in title.lower():
        return False
    return True


_nlp = None
_nlp_has_ner: bool = False


def is_ner_available() -> bool:
    """Check if the spaCy NER pipeline is currently loaded and operational."""
    global _nlp, _nlp_has_ner
    if _nlp is None:
        get_nlp()
    return _nlp_has_ner


def get_nlp():
    global _nlp, _nlp_has_ner
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            _nlp_has_ner = True
            logger.info("Loaded spaCy 'en_core_web_sm' model with full NER capability.")
        except Exception as e:
            logger.critical(
                "CRITICAL: spaCy 'en_core_web_sm' model not found! Fallback to blank english pipeline with zero NER capability. "
                "Run 'python -m spacy download en_core_web_sm' to restore full NER capability. Error: %s",
                e,
            )
            _nlp = spacy.blank("en")
            _nlp_has_ner = False
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
    """
    Administrative only if the block STARTS with an admin phrase, or is a short block (<25 words)
    that contains one. Searching whole long blocks silently deleted real content, e.g. a networking
    paragraph that mentions an "acknowledgement".
    """
    low = text.lower().strip()
    head = re.sub(r"^[^a-z]+", "", low)
    if any(head.startswith(p) for p in ADMIN_PHRASES):
        return True
    return len(low.split()) < 25 and any(p in low for p in ADMIN_PHRASES)


def heading_like(text: str) -> bool:
    t = norm(text)
    if not t or len(t) > 180:
        return False
    if re.match(r"(?i)^(?:chapter\s+)?(?:[ivxlc]+|\d+(?:\.\d+)*)[.)]?\s+", t):
        return True
    if t.isupper() and 2 <= len(t.split()) <= 18:
        return True
    return len(t.split()) <= 9 and not t.endswith((".", "?", "!", ";", ":"))


def discover_local_names(blocks: List[Dict[str, Any]], nlp, known_names: List[str] | None = None, title: str = "") -> List[str]:
    global _nlp, _nlp_has_ner
    has_ner = _nlp_has_ner or (nlp is not None and getattr(nlp, "has_pipe", lambda p: False)("ner"))
    if not has_ner:
        logger.warning(
            "NER-based name redaction is degraded: spaCy NER model ('en_core_web_sm') is not loaded. "
            "Proceeding with regex and front-matter heuristic name extraction."
        )

    names = set(n for n in (known_names or []) if looks_like_name(n))
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

        # Run NER on the text if NER component is available
        if has_ner:
            doc = nlp(text[:8000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = norm(ent.text)
                    # Filter out single words or non-name entities
                    if len(name) >= 5 and looks_like_name(name, title):
                        names.add(name)

        # Also inspect line-by-line for explicit role matches and ALL-CAPS names
        for line in text.splitlines():
            line_str = norm(line)
            if ROLE_RE.search(line_str):
                if has_ner:
                    line_doc = nlp(line_str)
                    for ent in line_doc.ents:
                        if ent.label_ == "PERSON":
                            name = norm(ent.text)
                            if looks_like_name(name, title):
                                names.add(name)
                # Regex heuristic extraction for explicit role lines (works even when NER is degraded)
                candidate = ROLE_RE.sub("", line_str)
                candidate = re.sub(r"^[\s:=-]+", "", candidate).strip()
                candidate = re.sub(r"(?i)^(?:prof(?:\.|essor)?|dr\.|mr\.|mrs\.|ms\.)\s*", "", candidate).strip()
                candidate = re.split(r"[\(\[\{]|(?:roll\s*no)|(?:prn)", candidate, flags=re.I)[0].strip()
                if looks_like_name(candidate, title):
                    names.add(candidate)
            elif is_front_zone and line_str.isupper() and 2 <= len(line_str.split()) <= 4:
                # Bare uppercase student name in front-matter zone
                if looks_like_name(line_str, title) and not any(h in line_str for h in HEADING_EXCLUSIONS):
                    names.add(line_str.title())
                    names.add(line_str)

    return sorted(names, key=lambda x: (-len(x), x.lower()))


def redact_text(text: str, local_names: List[str]) -> Tuple[str, Dict[str, int]]:
    global _nlp, _nlp_has_ner
    if _nlp is None:
        get_nlp()
    if not _nlp_has_ner and not getattr(redact_text, "_warned_degraded", False):
        logger.warning(
            "NER-based name redaction is degraded: spaCy NER model is unavailable. "
            "Applying regex-based redactions (emails, phones, IDs, credentials)."
        )
        setattr(redact_text, "_warned_degraded", True)

    text = norm(text)
    counts = {"email": 0, "phone": 0, "credential": 0, "student_id": 0, "local_name": 0}

    def _sub_email(m):
        counts["email"] += 1
        return "[EMAIL]"

    def _sub_phone(m):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if not 10 <= len(digits) <= 13 or re.search(r"\d\.\d", raw):
            return raw  # a metrics table or decimal series, not a phone number
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

    front = data.get("metadata") or meta.get("metadata") or {}
    seeds = list(front.get("authors") or []) + ([front["supervisor"]] if front.get("supervisor") else [])
    local_names = discover_local_names(raw_blocks, nlp, known_names=seeds, title=str(front.get("title") or ""))
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

    # extracted front-matter metadata holds the real author/supervisor names: never carry them forward
    doc_out = {k: v for k, v in meta.items() if k != "source_path"}
    if isinstance(doc_out.get("metadata"), dict):
        doc_out["metadata"] = {k: v for k, v in doc_out["metadata"].items() if k not in ("authors", "supervisor")}

    return {
        "document": doc_out,
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
