from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .encoder import MultimodalEncoder, l2_normalize


@dataclass(frozen=True)
class CoverageFacet:
    axis: str
    name: str
    prompts: tuple[str, ...]
    quota: int


@dataclass(frozen=True)
class SelectionReason:
    index: int
    axis: str
    label: str
    score: float
    rank_within_label: int


# Quotas sum to exactly 1,000. They are curation targets, not claimed labels.
DEFAULT_COVERAGE_FACETS: tuple[CoverageFacet, ...] = (
    CoverageFacet("environment", "modern office", ("professional fashion photo inside a modern office", "person in an office interior", "corporate workplace fashion"), 80),
    CoverageFacet("environment", "urban street", ("street-style fashion on an urban sidewalk", "person walking on a city street", "downtown fashion photo"), 80),
    CoverageFacet("environment", "park", ("fashion photo in a green park", "person near a park bench", "outfit photographed in a garden"), 80),
    CoverageFacet("environment", "home", ("casual fashion photo inside a home", "person in a living room", "outfit photographed in a home interior"), 80),
    CoverageFacet("clothing", "formal", ("formal business attire with blazer or suit", "professional button-down outfit", "elegant formal clothing"), 80),
    CoverageFacet("clothing", "casual", ("casual weekend outfit with t-shirt or hoodie", "relaxed everyday clothing", "casual streetwear"), 80),
    CoverageFacet("clothing", "outerwear", ("person wearing outerwear", "fashion photo of a coat or raincoat", "layered jacket outfit"), 80),
    CoverageFacet("color", "black", ("person wearing black clothing", "black fashion outfit"), 40),
    CoverageFacet("color", "white", ("person wearing white clothing", "white fashion outfit"), 40),
    CoverageFacet("color", "red", ("person wearing red clothing", "red fashion outfit"), 40),
    CoverageFacet("color", "blue", ("person wearing blue clothing", "blue fashion outfit"), 40),
    CoverageFacet("color", "yellow", ("person wearing yellow clothing", "bright yellow fashion outfit"), 40),
    CoverageFacet("color", "green", ("person wearing green clothing", "green fashion outfit"), 40),
    CoverageFacet("color", "orange", ("person wearing orange clothing", "orange fashion outfit"), 40),
    CoverageFacet("color", "purple", ("person wearing purple clothing", "purple fashion outfit"), 40),
    CoverageFacet("color", "pink", ("person wearing pink clothing", "pink fashion outfit"), 40),
    CoverageFacet("color", "brown", ("person wearing brown clothing", "brown fashion outfit"), 40),
    CoverageFacet("color", "beige", ("person wearing beige clothing", "beige or cream fashion outfit"), 40),
)


def encode_facet_vectors(encoder: MultimodalEncoder, facets: Sequence[CoverageFacet]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for facet in facets:
        prompt_vectors = encoder.encode_texts(facet.prompts)
        vectors.append(l2_normalize(prompt_vectors.mean(axis=0, keepdims=True))[0])
    return np.stack(vectors).astype(np.float32)


def allocate_unique_by_quota(
    scores: np.ndarray,
    facets: Sequence[CoverageFacet],
) -> list[SelectionReason]:
    """Round-robin allocation prevents early facets from consuming every strong image."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(facets):
        raise ValueError("scores must have shape [images, facets]")
    required = sum(facet.quota for facet in facets)
    if scores.shape[0] < required:
        raise ValueError(f"Need at least {required} candidates, received {scores.shape[0]}")

    rankings = [np.argsort(-scores[:, column], kind="stable") for column in range(len(facets))]
    pointers = [0] * len(facets)
    counts = [0] * len(facets)
    used: set[int] = set()
    selected: list[SelectionReason] = []
    while len(selected) < required:
        made_progress = False
        for column, facet in enumerate(facets):
            if counts[column] >= facet.quota:
                continue
            ranking = rankings[column]
            while pointers[column] < len(ranking) and int(ranking[pointers[column]]) in used:
                pointers[column] += 1
            if pointers[column] >= len(ranking):
                raise ValueError(f"Could not fill quota for {facet.axis}:{facet.name}")
            image_index = int(ranking[pointers[column]])
            selected.append(
                SelectionReason(
                    index=image_index,
                    axis=facet.axis,
                    label=facet.name,
                    score=round(float(scores[image_index, column]), 6),
                    rank_within_label=pointers[column] + 1,
                )
            )
            used.add(image_index)
            counts[column] += 1
            pointers[column] += 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("Coverage allocation stalled")
    return selected


def facet_report(scores: np.ndarray, facets: Sequence[CoverageFacet], paths: Sequence[str], top_n: int = 8) -> list[dict]:
    rows: list[dict] = []
    for column, facet in enumerate(facets):
        order = np.argsort(-scores[:, column], kind="stable")[:top_n]
        rows.append(
            {
                **asdict(facet),
                "score_summary": {
                    "maximum": round(float(scores[:, column].max()), 6),
                    "median": round(float(np.median(scores[:, column])), 6),
                },
                "top_examples": [
                    {"path": paths[int(index)], "score": round(float(scores[int(index), column]), 6)}
                    for index in order
                ],
            }
        )
    return rows
