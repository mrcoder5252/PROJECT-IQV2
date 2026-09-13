# Project: Project-IQ

Project-IQ is a modular AI backend system that analyzes past final-year student project reports, abstracts, and presentations to detect research and implementation gaps, generate novel project ideas grounded in past student work, and recommend the best project domains for new student teams based on their skills.

## Architecture

Project-IQ is organized as an end-to-end modular pipeline with 7 core subsystems:
1. **Ingestion & Extraction (`src/extract.py`)**: Multi-format document parser supporting PDF, DOCX, PPTX, and DOC (via LibreOffice headless). Computes SHA-256 binary hash and generates stable content-derived document IDs (`doc_{sha256[:12]}`) with page/slide text and metadata extraction.
2. **Privacy & Sanitization (`src/clean.py`)**: Administrative front-matter removal (certificates, declarations, acknowledgements, TOC) and granular PII redaction (`[EMAIL]`, `[PHONE]`, `[STUDENT_ID]`, `[NAME]`, `[REDACTED_CREDENTIAL]`) using spaCy NER and regex patterns while strictly preserving IEEE `[1]` and APA `(Author, Year)` citations and reference lists.
3. **Structure-Aware Chunking (`src/chunk_for_rag.py`)**: Section boundary aware token chunker using `tiktoken` (`cl100k_base`) with a target of 450 tokens and a strict ceiling of $\le 520$ tokens. Implements 6-word shingle Jaccard near-duplicate detection ($\ge 0.60$) and Union-Find clustering for canonical chunk election.
4. **Vector Storage & Hybrid Retrieval (`src/vector_store.py`)**: Persistent ChromaDB store (`project_iq_chunks`) utilizing `all-MiniLM-L6-v2` embeddings (384-dim) combined with lexical/BM25 and section-filtered hybrid retrieval.
5. **Research Gap & Novel Ideation Engine (`src/gap_analyzer.py`)**: Systematic limitation extractor scanning `FUTURE SCOPE`, `CONCLUSION`, `RESULTS`, and `METHODOLOGY` sections. Grounded project idea generation via `google-genai` SDK (`gemini-2.5-flash`) with a dynamic, deterministic offline synthesis engine as fallback.
6. **Team Expertise Matcher (`src/team_matcher.py`)**: Evaluates student profiles against a multi-domain taxonomy using hybrid semantic embedding similarity + keyword overlap, generating readiness tiers and skill gap recommendations.
7. **Pipeline Orchestrator (`run_pipeline.py`) & Dashboard (`web/app.py`)**: CLI runner executing the end-to-end ingestion-to-report pipeline with JSON report outputs, complemented by an interactive 4-tab Streamlit dashboard.

---

## Feature Inventory

Every requirement from the Survey phase is mapped below with its assigned milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Multi-format PDF Parsing | PyMuPDF text extraction with PyTesseract OCR fallback for scanned pages | M1 | Survey |
| 2 | Multi-format Office Parsing | DOCX (python-docx), PPTX (python-pptx), DOC (LibreOffice headless fallback) | M1 | Survey |
| 3 | Binary SHA-256 Hashing | Streamed SHA-256 calculation for document integrity and provenance | M1 | Survey |
| 4 | Deterministic Document IDs | Stable IDs formatted as `doc_{sha256[:12]}` derived strictly from content | M1 | Survey |
| 5 | Metadata Extraction | Front-page block parsing for Title, Author(s), Year, Supervisor, Category | M1 | Survey |
| 6 | Extraction Manifest | Generation of `_manifest.json` cataloging all extracted document records | M1 | Survey |
| 7 | Sample Corpus Generation | Utility script generating realistic PDF, DOCX, PPTX test documents | M1 | Survey |
| 8 | Administrative Front-Matter Stripping | Removal of certificates, declarations, acknowledgements, plagiarism reports | M2 | Survey |
| 9 | Multi-Type PII Redaction | Regex and heuristic masking of emails, phone numbers, student IDs, credentials | M2 | Survey |
| 10 | Context-Aware Name Redaction | Identification of student/guide names via spaCy NER bounded to front zones | M2 | Survey |
| 11 | Scholarly Citation Preservation | Protection of IEEE brackets `[1]` and APA `(Author, Year)` from redaction | M2 | Survey |
| 12 | Bibliography Section Protection | Retention of `REFERENCES` and `BIBLIOGRAPHY` blocks for citation analysis | M2 | Survey |
| 13 | Section Boundary Detection | Structural splitting by standard academic headers (Abstract, Intro, Methods, etc.) | M3 | Survey |
| 14 | tiktoken Bounded Chunking | `cl100k_base` encoding with target 450 tokens and strict hard limit $\le 520$ | M3 | Survey |
| 15 | Sliding Window Overlap | Inter-chunk context continuity (80 token overlap) bounded within sections | M3 | Survey |
| 16 | Shingle Duplicate Detection | 6-word shingle hashing with Jaccard coefficient $\ge 0.60$ for duplicate clustering | M3 | Survey |
| 17 | Canonical Chunk Election | Union-Find clustering selecting canonical chunks based on document hierarchy | M3 | Survey |
| 18 | Persistent ChromaDB Collection | Storage in `data/chroma_db` under collection `project_iq_chunks` | M4 | Survey |
| 19 | SentenceTransformer Embeddings | 384-dimensional dense vectors using `all-MiniLM-L6-v2` | M4 | Survey |
| 20 | Metadata-Filtered Dense Query | Cosine similarity search filtered by section, category, and document ID | M4 | Survey |
| 21 | Hybrid Section Retrieval | Combination of dense semantic similarity and lexical keyword/BM25 ranking | M4 | Survey |
| 22 | Research Gap Extraction | Regex-driven scanning of limitation, challenge, and future work markers | M5 | Survey |
| 23 | Grounded Gemini Ideation | API prompt synthesis generating novel project proposals citing past gaps | M5 | Survey |
| 24 | Deterministic Offline Ideation | Fallback synthesizer generating validated structured JSON proposals offline | M5 | Survey |
| 25 | Structured Proposal Schema | Enforcing Title, Problem, Methodology, Past Project Gaps, and Feasibility | M5 | Survey |
| 26 | Multi-Domain Engineering Taxonomy | Taxonomy of 7 distinct computer science and engineering domains | M6 | Survey |
| 27 | Hybrid Team Compatibility Scoring | 60% semantic embedding + 40% keyword overlap composite matching score | M6 | Survey |
| 28 | Readiness Tiering & Learning Advisor | Categorization (High/Moderate/Low) with actionable skill gap recommendations | M6 | Survey |
| 29 | Pipeline CLI Orchestrator | `run_pipeline.py` managing end-to-end data lifecycle with CLI flags | M7 | Survey |
| 30 | Execution Summary Report | Output of `pipeline_summary.json` with counts, timing, and gap stats | M7 | Survey |
| 31 | Streamlit Verification | Verification of `web/app.py` against live indexed ChromaDB and gap engine | M7 | Survey |
| 32 | Opaque-Box E2E Test Suite | Comprehensive multi-tier test suite in `tests/` verifying all 7 requirements | E2E Track | Survey |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven test suite (Tiers 1-4) in `tests/` + `TEST_READY.md` | none | COMPLETED |
| 1 | Ingestion & Extraction | `src/extract.py`: SHA-256 hashing, `doc_{sha256[:12]}` IDs, metadata extraction, `scripts/generate_sample_data.py` | none | COMPLETED |
| 2 | Privacy & Cleaning | `src/clean.py`: Front-matter stripping, PII redaction, citation preservation | M1 | COMPLETED |
| 3 | Section Chunking | `src/chunk_for_rag.py`: `cl100k_base`, max 520 tokens, 6-shingle Jaccard deduplication | M2 | COMPLETED |
| 4 | ChromaDB & Retrieval | `src/vector_store.py`: Persistent storage, hybrid dense + lexical retrieval | M3 | COMPLETED |
| 5 | Gap Analysis & Ideation | `src/gap_analyzer.py`: Gaps mining, Gemini API + deterministic offline fallback | M4 | COMPLETED |
| 6 | Team Profiling & Matcher | `src/team_matcher.py`: 7-domain taxonomy, composite scoring, readiness tiering | M4 | COMPLETED |
| 7 | Full Pipeline & Verification | `run_pipeline.py` end-to-end execution, 100% E2E test pass, adversarial hardening, Docker & a11y polish | M1-M6, E2E | COMPLETED |

---

## Interface Contracts

### 1. Ingestion (`src/extract.py`) $\to$ Cleaning (`src/clean.py`)
- **Caller**: `run_pipeline.py` or standalone scripts
- **Interface**: `extract_all(raw_dir: Path, output_dir: Path) -> List[Dict[str, Any]]`
- **Output Record Schema (`ExtractedDocument`)**:
  ```python
  {
      "doc_id": "doc_a1b2c3d4e5f6",        # 16-character string starting with doc_
      "source_file": "report.pdf",
      "sha256": "a1b2c3d4e5f6...",          # Full 64-char hex string
      "category": "project_report",
      "extracted_at": "2026-09-13T12:00:00Z",
      "metadata": {
          "title": str,
          "authors": List[str],
          "year": Optional[str],
          "supervisor": Optional[str]
      },
      "blocks": [
          {
              "block_id": "doc_a1b2c3d4e5f6_b0001",
              "page": 1,
              "text": "...",
              "char_count": 120
          }
      ]
  }
  ```

### 2. Cleaning (`src/clean.py`) $\to$ Chunking (`src/chunk_for_rag.py`)
- **Caller**: `run_pipeline.py`
- **Interface**: `clean_all(extracted_dir: Path, output_dir: Path) -> List[Dict[str, Any]]`
- **Output Record Schema (`CleanedDocument`)**:
  ```python
  {
      "doc_id": "doc_a1b2c3d4e5f6",
      "source_file": "report.pdf",
      "category": "project_report",
      "redaction_counts": {
          "email": 1,
          "phone": 0,
          "student_id": 2,
          "name": 4,
          "credential": 0
      },
      "blocks": [
          {
              "block_id": "doc_a1b2c3d4e5f6_b0001",
              "page": 1,
              "cleaned_text": "...",
              "is_front_matter": False
          }
      ]
  }
  ```

### 3. Chunking (`src/chunk_for_rag.py`) $\to$ Vector Store (`src/vector_store.py`)
- **Caller**: `run_pipeline.py`
- **Interface**: `chunk_all(cleaned_dir: Path, output_file: Path) -> List[Dict[str, Any]]`
- **Output Chunk Record Schema (`ChunkRecord`)**:
  ```python
  {
      "chunk_id": "doc_a1b2c3d4e5f6_chunk_001",
      "document_id": "doc_a1b2c3d4e5f6",
      "source_file": "report.pdf",
      "category": "project_report",
      "section": "METHODOLOGY",
      "text": "...",
      "token_count": 412,                   # Strict constraint: <= 520 tokens
      "is_canonical": True,
      "duplicate_group_id": "grp_001",
      "has_citation": True,
      "has_numbers": True
  }
  ```

### 4. Vector Store (`src/vector_store.py`) $\to$ Gap Analyzer (`src/gap_analyzer.py`)
- **Caller**: `run_pipeline.py` / `gap_analyzer.py` / `web/app.py`
- **Interface**:
  - `index_chunks(chunks_file: Path, reset: bool = False) -> int`
  - `query_similar(query: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]`
  - `query_by_section(query: str, section: str, n_results: int = 5) -> List[Dict[str, Any]]`
  - `query_hybrid(query: str, section: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]]`

### 5. Gap Analyzer (`src/gap_analyzer.py`) $\to$ Reports
- **Caller**: `run_pipeline.py`
- **Interface**:
  - `extract_project_gaps(store: ProjectIQVectorStore) -> List[Dict[str, Any]]`
  - `generate_novel_projects(gaps: List[Dict[str, Any]], domain: str = "all") -> List[Dict[str, Any]]`

### 6. Team Matcher (`src/team_matcher.py`) $\to$ Reports / UI
- **Caller**: `run_pipeline.py` / `web/app.py`
- **Interface**:
  - `evaluate_team(skills: List[str], interests: Optional[List[str]] = None) -> Dict[str, Any]`

---

## Code Layout

```
project_iq/
├── ORIGINAL_REQUEST.md         # Immutable user request specification
├── PROJECT.md                  # System architecture, milestones, and contracts
├── TEST_READY.md               # E2E test suite publication signal
├── README.md                   # Project overview and run instructions
├── run_pipeline.py             # CLI orchestrator executing stages end-to-end
├── src/
│   ├── __init__.py             # Package marker
│   ├── extract.py              # Multi-format ingestion & SHA-256 stable IDs
│   ├── clean.py                # PII redaction & citation preservation
│   ├── chunk_for_rag.py        # Section-aware token chunking & shingle deduplication
│   ├── vector_store.py         # ChromaDB indexing & hybrid retrieval
│   ├── gap_analyzer.py         # Research gaps extraction & novel ideation engine
│   └── team_matcher.py         # Student team expertise & domain matching
├── scripts/
│   └── generate_sample_data.py # Sample multi-format academic documents generator
├── web/
│   └── app.py                  # Streamlit interactive dashboard
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py       # Unit tests for multi-format extraction and hashing
│   ├── test_cleaning.py        # Unit tests for PII redaction & citation preservation
│   ├── test_chunking.py        # Unit tests for token chunking and shingle clustering
│   ├── test_vector_store.py    # Unit tests for ChromaDB and hybrid search
│   ├── test_gap_analyzer.py    # Unit tests for gap mining and proposal schemas
│   ├── test_team_matcher.py    # Unit tests for team profiling and domain scoring
│   ├── test_e2e_pipeline.py    # Multi-tier end-to-end pipeline verification tests
│   └── test_adversarial.py     # Adversarial boundary and edge-case tests
└── data/
    ├── raw/                    # Raw input documents (PDF, DOCX, PPTX)
    ├── extracted/              # Extracted document JSONs and manifest
    ├── cleaned/                # Sanitized document JSONs
    ├── chunks/                 # Token-bounded deduplicated chunks JSON
    ├── chroma_db/              # Persistent ChromaDB vector database
    └── reports/                # Output analysis JSONs and proposal reports
```
