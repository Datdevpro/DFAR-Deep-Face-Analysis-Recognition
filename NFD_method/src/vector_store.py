"""
FAISS vector store for face embeddings.

Index type: IndexFlatIP (inner product / cosine similarity).

Why IndexFlatIP with L2-normalised vectors?
  cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
  When ||a|| = ||b|| = 1 (both L2-normalised),
    cosine_similarity(a, b) = dot(a, b)
  FAISS inner-product search therefore directly ranks by cosine similarity.
  This avoids the overhead of IndexFlatL2 + manual distance conversion.

Index layout
  Each FAISS entry has an integer vector_id (0, 1, 2, …).
  metadata.json maps  vector_id → {employee_id, name, image_path}

Search aggregation (1:N mode)
  top-k raw results may contain multiple vectors from the same employee.
  We aggregate by employee_id (take max score per employee), then pick
  top-1 and top-2 employees to compute the identity margin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss  # type: ignore
import numpy as np

from src.utils import SearchResult


class FaceVectorStore:
    """
    FAISS-backed cosine-similarity store for 512-D face embeddings.

    Parameters
    ----------
    embedding_dim:
        Dimension of embeddings (default 512 for ArcFace).
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self._dim = embedding_dim
        # IndexFlatIP: exact inner-product search (cosine when L2-normalised)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        # metadata[vector_id] = {employee_id, name, image_path}
        self._metadata: dict[int, dict] = {}

    # ── Write API ─────────────────────────────────────────────────────────────

    def add_embeddings(
        self,
        employee_id: str,
        name: str,
        embeddings: list[np.ndarray],
        image_paths: Optional[list[str]] = None,
    ) -> int:
        """
        Add one or more embeddings for an employee.

        Parameters
        ----------
        employee_id:
            Unique identifier (e.g. "EMP001").
        name:
            Display name (e.g. "Nguyen Van A").
        embeddings:
            List of L2-normalised (512,) float32 arrays.
        image_paths:
            Optional list of source image paths (same length as embeddings).

        Returns
        -------
        int
            Number of vectors added.
        """
        if not embeddings:
            return 0

        if image_paths is None:
            image_paths = [""] * len(embeddings)

        matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 512)

        start_id = int(self._index.ntotal)
        self._index.add(matrix)

        for i, (emb, img_path) in enumerate(zip(embeddings, image_paths)):
            vid = start_id + i
            self._metadata[vid] = {
                "employee_id": employee_id,
                "name": name,
                "image_path": img_path,
            }

        return len(embeddings)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Write FAISS index and metadata JSON to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """Load FAISS index and metadata JSON from disk."""
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self._index = faiss.read_index(str(index_path))
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        # JSON keys are strings; convert back to int
        self._metadata = {int(k): v for k, v in raw.items()}

    @property
    def total_vectors(self) -> int:
        return int(self._index.ntotal)

    @property
    def enrolled_employees(self) -> set[str]:
        return {v["employee_id"] for v in self._metadata.values()}

    # ── Search API ────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search for the top-k most similar employees.

        Raw FAISS results (per-vector) are aggregated per employee
        by taking the maximum cosine score for each employee_id.

        Parameters
        ----------
        embedding:
            L2-normalised query vector, shape (512,).
        top_k:
            Number of *per-vector* nearest neighbours to retrieve
            before aggregation (should be >= 2× expected employees).

        Returns
        -------
        List[SearchResult]
            Sorted by descending score, one entry per unique employee.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query = embedding.astype(np.float32).reshape(1, -1)
        scores, ids = self._index.search(query, k)  # (1, k)

        # ── Aggregate per employee (take max score) ───────────────────────────
        emp_scores: dict[str, float] = {}
        emp_meta:   dict[str, dict]  = {}

        for score, vid in zip(scores[0], ids[0]):
            if vid < 0:
                continue  # FAISS pads with -1 when k > ntotal
            meta = self._metadata.get(int(vid))
            if meta is None:
                continue
            eid = meta["employee_id"]
            if eid not in emp_scores or score > emp_scores[eid]:
                emp_scores[eid] = float(score)
                emp_meta[eid] = meta

        # ── Build sorted result list ───────────────────────────────────────────
        results = [
            SearchResult(
                employee_id=eid,
                name=emp_meta[eid]["name"],
                score=emp_scores[eid],
                image_path=emp_meta[eid].get("image_path", ""),
            )
            for eid in emp_scores
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
