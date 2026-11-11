"""Tiny local RAG helpers (chunking + cosine search) used before the Knowledge Base exists."""
import re

import numpy as np


def chunk_by_heading(text, source):
    """Split a markdown document into one chunk per '## ' section."""
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks = []
    for part in re.split(r"\n(?=## )", text):
        body = part.strip()
        if body and not body.startswith("# "):
            heading = body.splitlines()[0].lstrip("# ").strip()
            chunks.append({"source": source, "title": title, "heading": heading,
                           "text": f"{title} - {body}"})
    return chunks


def chunk_fixed(text, source, size=400, overlap=50):
    """Split text into fixed-size character windows with overlap."""
    step = size - overlap
    return [{"source": source, "text": text[i:i + size]}
            for i in range(0, max(len(text) - overlap, 1), step)]


def top_k(query_vector, matrix, k=3):
    """Return (indices, scores) of the k most similar rows (vectors must be normalized)."""
    scores = np.asarray(matrix) @ np.asarray(query_vector)
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]
