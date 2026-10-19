"""
Project-IQ: Vector storage and hybrid section/metadata retrieval (ChromaDB).

Embeddings are L2-normalised and the collection uses cosine distance, so
``similarity = 1 - distance`` is a true cosine similarity in [0, 1].

Collection versioning: the old L2 collection (``project_iq_chunks``) is left
untouched; this version writes ``project_iq_chunks_cosine``. Re-index once with
``python -m src.vector_store --chunks data/chunks/chunks.jsonl`` (or run the pipeline).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.config import config
from src.embeddings import encode, get_embedder
from src.logger import get_logger

logger = get_logger("project_iq.vector_store")

DEFAULT_MODEL_NAME = config.embedding_model_name
DEFAULT_DB_PATH = config.chroma_dir
LEGACY_COLLECTION_NAME = "project_iq_chunks"
COLLECTION_NAME = "project_iq_chunks_cosine"


class ProjectIQVectorStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, model_name: str = DEFAULT_MODEL_NAME):
        self.db_path = Path(db_path).resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.client = chromadb.PersistentClient(path=str(self.db_path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Project-IQ Academic Project Chunks", "hnsw:space": "cosine"},
        )
        self._warn_if_legacy_only()
        logger.debug("Initialized ProjectIQVectorStore at %s with %d items.", self.db_path, self.collection.count())

    # ------------------------------------------------------------------ helpers
    def _warn_if_legacy_only(self) -> None:
        if self.collection.count() > 0:
            return
        try:
            names = {c.name if hasattr(c, "name") else str(c) for c in self.client.list_collections()}
        except Exception:
            return
        if LEGACY_COLLECTION_NAME in names:
            logger.warning(
                "Found an old L2-distance collection '%s' and an empty cosine collection. "
                "Re-index once: python -m src.vector_store --chunks %s",
                LEGACY_COLLECTION_NAME, config.chunks_file,
            )

    @property
    def model(self):
        return get_embedder(self.model_name)

    def drop_legacy_collection(self) -> bool:
        """Delete the old L2 collection (only call after the cosine index is rebuilt)."""
        try:
            self.client.delete_collection(LEGACY_COLLECTION_NAME)
            return True
        except Exception:
            return False

    # ----------------------------------------------------------------- indexing
    def index_chunks(self, chunks_file: str | Path, batch_size: int = 64, prune: bool = False) -> int:
        """
        Index canonical chunks. With ``prune=True`` any chunk already in the collection
        but absent from this file (stale ids from an earlier run) is removed.
        """
        fpath = Path(chunks_file).resolve()
        if not fpath.exists():
            logger.error("Chunks file not found: %s", fpath)
            raise FileNotFoundError(f"Chunks file not found: {fpath}")

        records: List[Dict[str, Any]] = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed chunk line in %s: %s", fpath.name, exc)

        if not records:
            logger.warning("No valid chunk records to index from %s.", fpath.name)
            return 0

        # Duplicates stay in chunks.jsonl (nothing is deleted) but only canonical chunks are embedded.
        to_index = [r for r in records if r.get("is_canonical", True)]
        logger.info("Indexing %d canonical chunks (from %d total chunks)...", len(to_index), len(records))

        total = 0
        for i in range(0, len(to_index), batch_size):
            batch = to_index[i:i + batch_size]
            self.collection.upsert(
                ids=[b["chunk_id"] for b in batch],
                documents=[b["text"] for b in batch],
                embeddings=encode([b["text"] for b in batch], self.model_name).tolist(),
                metadatas=[{
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
                    "duplicate_group_id": str(b.get("duplicate_group_id", "")),
                } for b in batch],
            )
            total += len(batch)
            logger.debug("Indexed %d/%d chunks...", total, len(to_index))

        if prune:
            keep = {b["chunk_id"] for b in to_index}
            stale = [cid for cid in self.collection.get(include=[])["ids"] if cid not in keep]
            for j in range(0, len(stale), 500):
                self.collection.delete(ids=stale[j:j + 500])
            if stale:
                logger.info("Pruned %d stale chunks from the collection.", len(stale))

        logger.info("COLLECTION READY: %d items in ChromaDB.", self.collection.count())
        return total

    # ---------------------------------------------------------------- retrieval
    @staticmethod
    def _build_where(section: Optional[str], category: Optional[str], department: Optional[str]) -> Optional[Dict[str, Any]]:
        conditions = []
        if section:
            conditions.append({"section": {"$eq": section.upper()}})
        if category:
            conditions.append({"category": {"$eq": category}})
        if department:
            conditions.append({"department": {"$eq": department}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        section: Optional[str] = None,
        category: Optional[str] = None,
        min_similarity: float = 0.0,
        department: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search. ``similarity`` is cosine similarity in [0, 1]."""
        clean_q = (query_text or "").strip()
        total = self.collection.count()
        if not clean_q or total == 0:
            return []

        try:
            results = self.collection.query(
                query_embeddings=encode([clean_q], self.model_name).tolist(),
                n_results=min(max(1, n_results), total),
                where=self._build_where(section, category, department),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Vector query failed: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        ids = (results or {}).get("ids") or [[]]
        for idx, cid in enumerate(ids[0]):
            dist = float(results["distances"][0][idx])
            sim = min(1.0, max(0.0, 1.0 - dist))
            if sim >= min_similarity:
                out.append({
                    "chunk_id": cid,
                    "text": results["documents"][0][idx],
                    "similarity": round(sim, 4),
                    "distance": round(dist, 4),
                    "metadata": results["metadatas"][0][idx],
                })
        return out

    def query_by_section(self, section: str, query_text: Optional[str] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """Chunks of one section; ranked by relevance when ``query_text`` is given, otherwise unranked."""
        total = self.collection.count()
        if total == 0:
            return []

        sec = section.upper()
        if query_text:
            return self.query_similar(query_text=query_text, n_results=n_results, section=sec)

        try:
            got = self.collection.get(
                where={"section": {"$eq": sec}},
                limit=min(max(1, n_results), total),
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Query by section failed: %s", exc)
            return []

        # similarity is not meaningful for an unranked fetch; 1.0 is kept for backwards compatibility.
        return [
            {"chunk_id": got["ids"][i], "text": got["documents"][i], "similarity": 1.0, "metadata": got["metadatas"][i]}
            for i in range(len(got.get("ids") or []))
        ]

    def count(self) -> int:
        return self.collection.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index chunks into persistent ChromaDB (cosine).")
    parser.add_argument("--chunks", default=str(config.chunks_file), help="Path to chunks.jsonl")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="ChromaDB directory")
    parser.add_argument("--prune", action="store_true", help="Remove indexed chunks that are no longer in the file")
    parser.add_argument("--drop-legacy", action="store_true", help="Delete the old L2 collection after indexing")
    args = parser.parse_args()

    store = ProjectIQVectorStore(db_path=args.db)
    indexed = store.index_chunks(args.chunks, prune=args.prune)
    logger.info("Total indexed into %s: %d chunks.", args.db, indexed)
    if args.drop_legacy and store.drop_legacy_collection():
        logger.info("Dropped legacy L2 collection.")


if __name__ == "__main__":
    main()
