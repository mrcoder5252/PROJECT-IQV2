"""
Project-IQ: Conversational RAG Chat & Question Answering Engine
Enables natural language querying across the entire historical college project corpus,
providing grounded, cited, and comprehensive answers using either the Google Gemini API
or a rich local contextual synthesis engine.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional

from src.config import config
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore

logger = get_logger("project_iq.rag_chat")

RAG_SYSTEM_PROMPT = """You are Project-IQ Academic Advisor, an elite AI research assistant specialized in undergraduate engineering capstone and final-year projects.
You have access to historical student project reports, black books, presentation slide decks, and synopses across multiple departments (CS_1, CS_2, DS_1, DS_2).

Your job is to answer the user's question with high academic rigor, technical depth, and specific grounded citations to the retrieved project documents.

Guidelines:
1. Grounding: Rely strictly on the provided Context Chunks. When citing facts, mention the source file name, department, and group ID.
2. Structure: Use clear markdown with bold headers, bullet points, and technical details (e.g. algorithms, frameworks, metrics).
3. If asked for recommendations or gaps: Highlight limitations mentioned in past projects and propose concrete, innovative extensions.
4. If the provided context does not contain enough information to answer completely, acknowledge what is in the repository and offer reasonable engineering principles based on the available data.
"""

class RAGChatEngine:
    def __init__(self, vector_store: Optional[ProjectIQVectorStore] = None, api_key: Optional[str] = None):
        self.vector_store = vector_store or ProjectIQVectorStore()
        self.api_key = api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._gemini_client = None
        if self.api_key:
            self._init_gemini(self.api_key)

    def _init_gemini(self, api_key: str):
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            self.api_key = api_key
            logger.info("Initialized Google Gemini client for RAG conversational chat.")
        except Exception as e:
            logger.warning(f"Could not initialize google.genai: {e}. Falling back to local synthesis.")
            self._gemini_client = None

    def set_api_key(self, api_key: str):
        """Dynamically configure API key from dashboard UI."""
        self._init_gemini(api_key)

    def retrieve_context(
        self,
        query: str,
        n_results: int = 6,
        department_filter: Optional[str] = None,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant canonical chunks from ChromaDB."""
        sec = None if section_filter == "ALL" else section_filter
        dept = None if department_filter == "ALL" else department_filter

        results = self.vector_store.query_similar(
            query_text=query,
            n_results=n_results * 2 if dept else n_results,
            section=sec
        )

        if dept:
            results = [r for r in results if r["metadata"].get("department") == dept]

        return results[:n_results]

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        department_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        n_results: int = 6
    ) -> Dict[str, Any]:
        """
        Processes a user question, retrieves relevant context, and synthesizes
        a grounded answer with citations.
        """
        chat_history = chat_history or []
        retrieved_chunks = self.retrieve_context(
            query=question,
            n_results=n_results,
            department_filter=department_filter,
            section_filter=section_filter
        )

        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant project documents in the institutional repository matching your question. Please try rephrasing with different keywords or selecting 'ALL' in the department/section filters.",
                "sources": [],
                "engine_used": "no_context"
            }

        # Build context prompt
        context_blocks = []
        sources = []
        for idx, r in enumerate(retrieved_chunks, 1):
            meta = r["metadata"]
            src_file = meta.get("source_file", "Unknown Document")
            dept = meta.get("department", "General")
            grp = meta.get("group_id", "General")
            sec = meta.get("section", "General")
            sim = r.get("similarity", 0.0)
            text = r.get("text", "")

            context_blocks.append(f"--- Document [{idx}]: {src_file} ({dept}, {grp}) | Section: {sec} ---\n{text}\n")
            sources.append({
                "index": idx,
                "source_file": src_file,
                "department": dept,
                "group_id": grp,
                "section": sec,
                "similarity_score": round(sim, 3),
                "excerpt": text[:260] + ("..." if len(text) > 260 else "")
            })

        context_str = "\n".join(context_blocks)

        # Build conversation history block
        history_blocks = []
        if chat_history:
            history_blocks.append("### Previous Conversation Context:")
            for h in chat_history[-4:]:
                role = "User" if h.get("role") == "user" else "Advisor"
                history_blocks.append(f"{role}: {h.get('content', '')}")
            history_blocks.append("")
        hist_str = "\n".join(history_blocks)

        # Attempt Gemini Cloud Generation if Client is Ready
        if self._gemini_client:
            try:
                prompt = (
                    f"{RAG_SYSTEM_PROMPT}\n\n"
                    f"### Context Chunks from College Project Repository:\n{context_str}\n\n"
                    f"{hist_str}"
                    f"### Current User Question:\n{question}\n\n"
                    f"Please provide a comprehensive, well-structured answer with markdown headings and exact citations to Document [1], [2], etc."
                )
                response = self._gemini_client.models.generate_content(
                    model=config.llm_model_name,
                    contents=prompt
                )
                answer_text = response.text if hasattr(response, "text") else str(response)
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "engine_used": f"Gemini ({config.llm_model_name})"
                }
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Falling back to local synthesis engine.")

        # High-Fidelity Local Synthesizer Fallback
        local_answer = self._synthesize_local(question, retrieved_chunks, sources)
        return {
            "answer": local_answer,
            "sources": sources,
            "engine_used": "Local High-Fidelity RAG Synthesizer"
        }

    def _synthesize_local(self, question: str, chunks: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> str:
        """
        Structured local synthesis engine that extracts key findings, methodology patterns,
        limitations, technologies, metrics, and citations without needing an external cloud API.
        """
        # Categorize chunks by section
        methodology_points = []
        future_points = []
        result_points = []
        general_points = []

        known_techs = [
            "ResNet", "VGG", "YOLO", "CNN", "LSTM", "RNN", "BERT", "Transformer",
            "SVM", "Random Forest", "Naive Bayes", "KNN", "XGBoost", "Decision Tree",
            "K-Means", "OpenCV", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn",
            "Flask", "Django", "FastAPI", "React", "Firebase", "MongoDB", "MySQL",
            "Arduino", "Raspberry Pi", "ESP32", "Ethereum", "Solidity", "Blockchain"
        ]
        found_techs = set()
        found_metrics = []

        for c, s in zip(chunks, sources):
            meta = c["metadata"]
            text = c.get("text", "")
            sec = meta.get("section", "").upper()
            ref_tag = f"`[{s['source_file']} ({s['department']}, {s['group_id']})]`"

            # Check for known technologies
            for tech in known_techs:
                if re.search(r"\b" + re.escape(tech) + r"\b", text, re.IGNORECASE):
                    found_techs.add(tech)

            # Check for accuracy or metric statements
            pcts = re.findall(r"\b\d+(?:\.\d+)?%\s*(?:accuracy|f1|precision|recall|loss)?", text, re.IGNORECASE)
            for p in pcts:
                if len(found_metrics) < 5 and p.strip() not in found_metrics:
                    found_metrics.append(p.strip())

            # Extract key sentences (avoid trivial short ones)
            sentences = [
                st.strip() for st in re.split(r"[.!?]+\s+", text)
                if len(st.strip().split()) >= 8
            ]

            for sent in sentences[:2]:
                entry = f"- {sent}. {ref_tag}"
                if any(k in sec for k in ["METHODOLOGY", "ARCHITECTURE", "PROPOSED", "IMPLEMENTATION"]):
                    methodology_points.append(entry)
                elif any(k in sec for k in ["FUTURE", "LIMITATION", "SCOPE", "CONCLUSION"]):
                    future_points.append(entry)
                elif any(k in sec for k in ["RESULT", "EVALUATION", "DISCUSSION"]):
                    result_points.append(entry)
                else:
                    general_points.append(entry)

        sections_out = []
        sections_out.append(f"### 📋 Academic Synthesis on: *\"{question}\"*\n")
        sections_out.append(
            f"Based on retrieval across **{len(chunks)} relevant project sections** from the institutional repository "
            f"across departments ({', '.join(set(s['department'] for s in sources))}), here is the synthesized intelligence:\n"
        )

        if found_techs or found_metrics:
            sections_out.append("#### ⚙️ Extracted Technical Stack & Empirical Benchmarks")
            if found_techs:
                tech_badges = " ".join([f"`{t}`" for t in sorted(found_techs)])
                sections_out.append(f"- **Identified Technologies & Models:** {tech_badges}")
            if found_metrics:
                sections_out.append(f"- **Reported Performance Metrics:** {', '.join(found_metrics)}")
            sections_out.append("")

        if methodology_points:
            sections_out.append("#### 🛠️ Implemented Systems, Architectures & Methods")
            sections_out.extend(methodology_points[:4])
            sections_out.append("")

        if result_points:
            sections_out.append("#### 📈 Experimental Findings & Evaluation Results")
            sections_out.extend(result_points[:3])
            sections_out.append("")

        if future_points:
            sections_out.append("#### 🔬 Reported Limitations & Unaddressed Research Gaps")
            sections_out.extend(future_points[:4])
            sections_out.append("")

        if not (methodology_points or future_points or result_points):
            sections_out.append("#### 🔍 Core Document Evidence")
            sections_out.extend(general_points[:5])
            sections_out.append("")

        sections_out.append("#### 💡 Strategic Guidance for Prospective Teams")
        sections_out.append(
            "- **Novelty Opportunity:** Address the limitations cited above by integrating edge-device quantization, robust validation metrics, or modern foundation models.\n"
            "- **Interdisciplinary Scope:** Cross-pollinate methodologies from multiple departments to elevate project depth beyond past college baselines."
        )

        return "\n".join(sections_out)
