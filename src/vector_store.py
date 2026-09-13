"""
Project-IQ: Vector Storage & Hybrid Section Retrieval Engine
Uses ChromaDB and SentenceTransformers (all-MiniLM-L6-v2) for indexing
and hybrid semantic/metadata-filtered retrieval.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.vector_store")

DEFAULT_MODEL_NAME = config.embedding_model_name
DEFAULT_DB_PATH = config.chroma_dir
COLLECTION_NAME = "project_iq_chunks"


class ProjectIQVectorStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, model_name: str = DEFAULT_MODEL_NAME):
        self.db_path = Path(db_path).resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Project-IQ Academic Project Chunks"}
        )
        logger.debug(f"Initialized ProjectIQVectorStore at {self.db_path} with {self.collection.count()} items.")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_chunks(self, chunks_file: str | Path, batch_size: int = 64) -> int:
        fpath = Path(chunks_file).resolve()
        if not fpath.exists():
            logger.error(f"Chunks file not found: {fpath}")
            raise FileNotFoundError(f"Chunks file not found: {fpath}")

        records: List[Dict[str, Any]] = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception as e:
                        logger.warning(f"Skipping malformed chunk line in {fpath.name}: {e}")

        if not records:
            logger.warning(f"No valid chunk records to index from {fpath.name}.")
            return 0

        # Filter to canonical chunks for optimal RAG indexing (duplicates still accessible via group ID)
        to_index = [r for r in records if r.get("is_canonical", True)]
        logger.info(f"Indexing {len(to_index)} canonical chunks (from {len(records)} total chunks)...")

        total_indexed = 0
        for i in range(0, len(to_index), batch_size):
            batch = to_index[i : i + batch_size]
            ids = [b["chunk_id"] for b in batch]
            documents = [b["text"] for b in batch]

            # Generate embeddings
            embeddings = self.model.encode(documents, show_progress_bar=False).tolist()

            metadatas = []
            for b in batch:
                metadatas.append({
                    "document_id": str(b.get("document_id", "")),
                    "source_file": str(b.get("source_file", "")),
                    "department": str(b.get("department", "General")),
                    "group_id": str(b.get("group_id", "General")),
                    "project_name": str(b.get("project_name", "Unknown")),
                    "category": str(b.get("category", "")),
                    "section": str(b.get("section", "BODY")),
                    "is_canonical": bool(b.get("is_canonical", True)),
                    "has_citation": bool(b.get("has_citation", False)),
                    "token_count": int(b.get("token_count", 0)),
                    "word_count": int(b.get("word_count", 0)),
                    "duplicate_group_id": str(b.get("duplicate_group_id", ""))
                })

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_indexed += len(batch)
            logger.debug(f"Indexed {total_indexed}/{len(to_index)} chunks...")

        logger.info(f"COLLECTION READY: {self.collection.count()} items in ChromaDB.")
        return total_indexed

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        section: Optional[str] = None,
        category: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        clean_q = (query_text or "").strip()
        if not clean_q or self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([clean_q], show_progress_bar=False).tolist()

        where_clause = {}
        conditions = []
        if section:
            conditions.append({"section": {"$eq": section.upper()}})
        if category:
            conditions.append({"category": {"$eq": category}})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        safe_n = min(n_results, max(1, self.collection.count()))
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=safe_n,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning(f"Vector query failed: {e}")
            return []

        out = []
        if results and results.get("ids") and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                cid = results["ids"][0][idx]
                doc = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                dist = results["distances"][0][idx]
                # Chroma uses L2 or cosine distance
                sim = 1.0 / (1.0 + dist)
                if sim >= min_similarity:
                    out.append({
                        "chunk_id": cid,
                        "text": doc,
                        "similarity": round(sim, 4),
                        "distance": round(dist, 4),
                        "metadata": meta
                    })
        return out

    def query_by_section(self, section: str, query_text: Optional[str] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        sec_upper = section.upper()
        if query_text:
            return self.query_similar(query_text=query_text, n_results=n_results, section=sec_upper)

        safe_n = min(n_results, max(1, self.collection.count()))
        try:
            got = self.collection.get(
                where={"section": {"$eq": sec_upper}},
                limit=safe_n,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.warning(f"Query by section failed: {e}")
            return []

        out = []
        if got and got.get("ids"):
            for idx in range(len(got["ids"])):
                out.append({
                    "chunk_id": got["ids"][idx],
                    "text": got["documents"][idx],
                    "similarity": 1.0,
                    "metadata": got["metadatas"][idx]
                })
        return out

    def count(self) -> int:
        return self.collection.count()


def main():
    parser = argparse.ArgumentParser(description="Index chunks into persistent ChromaDB.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Directory path for ChromaDB storage")
    args = parser.parse_args()

    store = ProjectIQVectorStore(db_path=args.db)
    indexed = store.index_chunks(args.chunks)
    logger.info(f"Total indexed into {args.db}: {indexed} chunks.")


if __name__ == "__main__":
    main()
