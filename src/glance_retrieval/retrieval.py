from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .encoder import MultimodalEncoder, l2_normalize
from .index_store import IndexStore
from .query import binding_components, binding_distractors, parse_query, prompt_variants


@dataclass(frozen=True)
class SearchResult:
    rank: int
    image_id: str
    path: str
    score: float
    global_score: float
    conjunction_score: float
    facet_scores: dict[str, float]


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(values.std())
    if scale < 1e-6:
        return np.zeros_like(values)
    return np.clip((values - float(values.mean())) / scale, -4.0, 4.0)


def _soft_min(matrix: np.ndarray, temperature: float = 0.35) -> np.ndarray:
    """Smooth AND: a low-scoring required facet pulls the result down."""
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    scaled = -matrix / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    log_mean_exp = maximum[:, 0] + np.log(np.exp(scaled - maximum).mean(axis=1))
    return -temperature * log_mean_exp


class FashionRetriever:
    def __init__(self, index_dir: Path, encoder: MultimodalEncoder):
        self.store = IndexStore(index_dir)
        self.encoder = encoder
        self.manifest, self.metadata, self.global_vectors, self.region_vectors, self.region_mask = self.store.load()
        if self.manifest.model_id != encoder.model_id:
            raise ValueError(
                f"Model mismatch: index uses {self.manifest.model_id!r}, query encoder uses {encoder.model_id!r}"
            )

    def _ensemble_text(self, prompts: Sequence[str]) -> np.ndarray:
        return l2_normalize(self.encoder.encode_texts(prompts).mean(axis=0, keepdims=True))[0]

    def search(self, text: str, k: int = 5, candidate_k: int = 100) -> list[SearchResult]:
        parsed = parse_query(text)
        full_query = self._ensemble_text((text, f"a fashion photo of {text}"))
        candidate_ids, _ = self.store.candidate_search(full_query, max(candidate_k, k), self.global_vectors)
        candidate_global = np.asarray(self.global_vectors[candidate_ids])
        raw_global = candidate_global @ full_query
        z_global = _standardize(raw_global)

        facet_columns: list[np.ndarray] = []
        facet_names: list[str] = []
        binding_count = len(parsed.bindings)
        binding_colors = tuple(color for color, _ in (binding_components(item) for item in parsed.bindings) if color)
        for kind, facets in (
            ("binding", parsed.bindings),
            ("scene", parsed.scenes),
            ("style", parsed.styles),
            ("action", parsed.actions),
        ):
            for facet in facets:
                query_vector = self._ensemble_text(prompt_variants(facet, kind))
                if kind == "binding":
                    regions = np.asarray(self.region_vectors[candidate_ids])
                    scores = regions @ query_vector
                    mask = np.asarray(self.region_mask[candidate_ids])
                    scores = np.where(mask, scores, -1e9)
                    phrase_raw = scores.max(axis=1)
                    _, garment = binding_components(facet)
                    garment_vector = self._ensemble_text((f"a person wearing a {garment}", f"a fashion photo of a {garment}", garment))
                    garment_scores = np.where(mask, regions @ garment_vector, -1e9)
                    garment_raw = garment_scores.max(axis=1)
                    distractors = binding_distractors(facet, binding_colors)
                    if distractors:
                        distractor_vectors = np.stack([
                            self._ensemble_text(prompt_variants(item, "binding")) for item in distractors
                        ])
                        # Strong evidence for a color on the wrong garment is an explicit counterexample.
                        distractor_scores = np.where(mask[:, :, None], regions @ distractor_vectors.T, -1e9)
                        # Compare within the same crop so color on trousers cannot satisfy a tie request.
                        local_margin = np.max(scores - np.max(distractor_scores, axis=2), axis=1)
                    else:
                        local_margin = np.zeros_like(phrase_raw)
                    raw = 0.55 * phrase_raw + 0.30 * garment_raw + 0.45 * local_margin
                else:
                    raw = candidate_global @ query_vector
                facet_columns.append(_standardize(raw))
                facet_names.append(f"{kind}:{facet}")

        if parsed.negatives:
            negative_vectors = [self._ensemble_text((item, f"a person wearing {item}")) for item in parsed.negatives]
            negative_raw = np.max(candidate_global @ np.stack(negative_vectors).T, axis=1)
            negative_penalty = np.maximum(_standardize(negative_raw), 0.0)
        else:
            negative_penalty = np.zeros(len(candidate_ids), dtype=np.float32)

        if facet_columns:
            facet_matrix = np.stack(facet_columns, axis=1)
            conjunction = _soft_min(facet_matrix)
            # Bound garment/color phrases deserve most weight; other facets preserve context.
            binding_mean = facet_matrix[:, :binding_count].mean(axis=1) if binding_count else np.zeros_like(conjunction)
            other_mean = facet_matrix[:, binding_count:].mean(axis=1) if binding_count < facet_matrix.shape[1] else np.zeros_like(conjunction)
            # Global similarity retrieves broadly; explicit facets dominate final precision.
            final = 0.15 * z_global + 0.55 * conjunction + 0.20 * binding_mean + 0.10 * other_mean - 0.20 * negative_penalty
        else:
            facet_matrix = np.empty((len(candidate_ids), 0), dtype=np.float32)
            conjunction = z_global
            final = z_global - 0.20 * negative_penalty

        order = np.argsort(-final, kind="stable")[:k]
        results: list[SearchResult] = []
        for rank, position in enumerate(order, start=1):
            image_index = int(candidate_ids[position])
            facet_scores = {name: round(float(facet_matrix[position, col]), 4) for col, name in enumerate(facet_names)}
            results.append(
                SearchResult(
                    rank=rank,
                    image_id=self.metadata[image_index]["image_id"],
                    path=self.metadata[image_index]["path"],
                    score=round(float(final[position]), 6),
                    global_score=round(float(z_global[position]), 6),
                    conjunction_score=round(float(conjunction[position]), 6),
                    facet_scores=facet_scores,
                )
            )
        return results
