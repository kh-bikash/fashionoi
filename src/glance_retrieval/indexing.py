from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import ImageRecord, load_region_images
from .encoder import MultimodalEncoder, l2_normalize
from .index_store import IndexStore, Manifest


def build_index(
    records: list[ImageRecord],
    encoder: MultimodalEncoder,
    output_dir: Path,
    build_faiss: bool = True,
    record_batch_size: int = 8,
    progress: Callable[[int, int, Path], None] | None = None,
) -> Manifest:
    if not records:
        raise ValueError("No images were selected for indexing")
    vectors_by_image: list[np.ndarray] = []
    metadata: list[dict] = []
    record_batch_size = max(1, record_batch_size)
    for start in range(0, len(records), record_batch_size):
        record_batch = records[start : start + record_batch_size]
        image_batches = [load_region_images(record) for record in record_batch]
        counts = [len(images) for images in image_batches]
        flat_images = [image for images in image_batches for image in images]
        flat_vectors = l2_normalize(encoder.encode_images(flat_images))
        if flat_vectors.ndim != 2 or flat_vectors.shape[0] != len(flat_images):
            raise ValueError("Encoder returned an unexpected image embedding shape")
        offset = 0
        for local_index, (record, count) in enumerate(zip(record_batch, counts)):
            vectors = flat_vectors[offset : offset + count]
            offset += count
            vectors_by_image.append(vectors)
            used_regions = list(record.regions[:count])
            metadata.append(
                {
                    "image_id": record.image_id,
                    "path": str(record.path),
                    "width": record.width,
                    "height": record.height,
                    "regions": [{"name": region.name, "box": list(region.box)} for region in used_regions],
                }
            )
            number = start + local_index + 1
            if progress:
                progress(number, len(records), record.path)

    dimension = int(vectors_by_image[0].shape[1])
    if any(v.shape[1] != dimension for v in vectors_by_image):
        raise ValueError("Encoder dimension changed during indexing")
    max_regions = max(v.shape[0] for v in vectors_by_image)
    region_vectors = np.zeros((len(records), max_regions, dimension), dtype=np.float32)
    region_mask = np.zeros((len(records), max_regions), dtype=np.bool_)
    for row, vectors in enumerate(vectors_by_image):
        region_vectors[row, : len(vectors)] = vectors
        region_mask[row, : len(vectors)] = True
    global_vectors = l2_normalize(region_vectors[:, 0, :])
    manifest = Manifest(
        version=1,
        model_id=encoder.model_id,
        dimension=dimension,
        image_count=len(records),
        max_regions=max_regions,
    )
    IndexStore(output_dir).write(manifest, metadata, global_vectors, region_vectors, region_mask, build_faiss)
    return manifest
