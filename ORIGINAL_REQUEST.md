# Original User Request

## 2026-09-13T08:58:56Z

Project-IQ is a modular AI backend system that analyzes past final-year student project reports, abstracts, and presentations to detect research and implementation gaps, generate novel project ideas grounded in past student work, and recommend the best project domains for new student teams based on their skills.

Working directory: C:/Users/mrcod/.gemini/antigravity/scratch/project_iq
Integrity mode: development

Execute the full implementation of Project-IQ in C:/Users/mrcod/.gemini/antigravity/scratch/project_iq:
1. Multi-format ingestion and extraction (PDF, DOCX, PPTX, DOC) with SHA256 and stable IDs.
2. Privacy & cleaning pipeline (PII redaction, admin front matter stripping, scholarly citation preservation).
3. Section-aware token chunking (tiktoken cl100k_base, max 520 tokens, near-duplicate shingle detection).
4. ChromaDB vector indexing and hybrid section retrieval.
5. Project gap analysis and novel ideation engine (Gemini API / fallback prompt synthesis).
6. Team expertise profiling and domain matcher.
7. Pipeline orchestrator run_pipeline.py and complete automated verification test suite.
