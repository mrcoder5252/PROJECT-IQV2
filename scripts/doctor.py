"""Environment check: python scripts/doctor.py  (prints what is missing; never prints secrets)."""
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ok = True


def check(label, passed, hint=""):
    global ok
    ok &= bool(passed)
    print(f"[{'OK' if passed else 'FAIL'}] {label}" + (f"  -> {hint}" if hint and not passed else ""))


check("Python >= 3.10", sys.version_info >= (3, 10), "install Python 3.10+")
for mod, pip in [("streamlit", "streamlit"), ("chromadb", "chromadb"), ("sentence_transformers", "sentence-transformers"),
                 ("spacy", "spacy"), ("fitz", "PyMuPDF"), ("docx", "python-docx"), ("pptx", "python-pptx"),
                 ("tiktoken", "tiktoken"), ("pandas", "pandas"), ("plotly", "plotly"), ("google.genai", "google-genai")]:
    try:
        importlib.import_module(mod)
        check(mod, True)
    except Exception:
        check(mod, False, f"pip install {pip}")
try:
    import spacy
    spacy.load("en_core_web_sm")
    check("spaCy model en_core_web_sm", True)
except Exception:
    check("spaCy model en_core_web_sm", False, "python -m spacy download en_core_web_sm  (needed for name redaction)")
check("Tesseract OCR (optional, scanned PDFs)", shutil.which("tesseract") is not None, "install Tesseract and add to PATH")
check("LibreOffice (optional, .doc files)", shutil.which("soffice") or shutil.which("libreoffice"), "install LibreOffice")

from src.config import config  # noqa: E402

raw = [p for p in config.raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".pptx", ".docx", ".doc"}]
check(f"Raw documents in {config.raw_dir} ({len(raw)} found)", len(raw) > 0, "copy your reports into data/raw/<DEPT>/<Group N>/...")
print(f"[{'OK' if config.gemini_api_key else 'INFO'}] Gemini key {'set' if config.gemini_api_key else 'not set (local fallback mode)'}")
print(f"[{'OK' if config.admin_passkey else 'INFO'}] Admin passkey {'set' if config.admin_passkey else 'not set (admin portal locked)'}")
sys.exit(0 if ok else 1)
