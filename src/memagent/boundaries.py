"""Detect semantic topic boundaries in a stream of conversation turns.

Approach: sliding-window centroid comparison. At each candidate split point,
compare the embedding centroid of the preceding window of turns against the
following window. When cosine similarity drops below a threshold, mark a
boundary.

Inspired by GAM's semantic-event-triggered partitioning (arxiv 2604.12285),
but uses local embeddings instead of an LLM call — cheaper and runs offline.
"""
from __future__ import annotations

import numpy as np

from .embedder import cosine_similarity, embed_batch
from .session_reader import Turn

DEFAULT_THRESHOLD = 0.72
DEFAULT_WINDOW = 3
MIN_SEGMENT_TURNS = 4


def split_at_boundaries(
    turns: list[Turn],
    threshold: float = DEFAULT_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    min_segment: int = MIN_SEGMENT_TURNS,
) -> list[list[Turn]]:
    """Split turns into coherent topic segments using sliding-window centroid similarity.

    Algorithm:
      1. Compute centroid-vs-centroid cosine similarity at every candidate
         split position (between two windows of `window` turns).
      2. A position is a boundary if (a) similarity < threshold AND (b) it is
         a local minimum (strictly lower than both immediate neighbors).
      3. Enforce min_segment turns between consecutive boundaries.

    Short conversations (< 2*window turns) return as a single segment.
    """
    if len(turns) < 2 * window:
        return [turns]

    embeddings = embed_batch([t.text for t in turns])

    # Compute the similarity series across all candidate positions
    sims: list[tuple[int, float]] = []
    for i in range(window, len(turns) - window):
        prev_centroid = np.mean(embeddings[i - window:i], axis=0)
        next_centroid = np.mean(embeddings[i:i + window], axis=0)
        sims.append((i, float(cosine_similarity(prev_centroid, next_centroid))))

    # Identify local minima below the threshold
    boundary_indices: list[int] = []
    for k in range(len(sims)):
        idx, sim = sims[k]
        if sim >= threshold:
            continue
        prev_sim = sims[k - 1][1] if k > 0 else float("inf")
        next_sim = sims[k + 1][1] if k < len(sims) - 1 else float("inf")
        if sim < prev_sim and sim < next_sim:
            if not boundary_indices or (idx - boundary_indices[-1]) >= min_segment:
                boundary_indices.append(idx)

    if not boundary_indices:
        return [turns]

    segments: list[list[Turn]] = []
    cuts = [0] + boundary_indices + [len(turns)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        segment = turns[start:end]
        if segment:
            segments.append(segment)
    return segments
