from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class MultimodalEncoder(Protocol):
    model_id: str

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


@dataclass
class OpenClipEncoder:
    """Lazy OpenCLIP adapter for Marqo FashionSigLIP or another HF Hub model."""

    model_id: str = "hf-hub:Marqo/marqo-fashionSigLIP"
    device: str | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Model dependencies are missing. Run `pip install -r requirements.txt`."
            ) from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(self.model_id)
        self._tokenizer = open_clip.get_tokenizer(self.model_id)
        self._model = self._model.to(self.device).eval()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        batches: list[np.ndarray] = []
        torch = self._torch
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack([self._preprocess(image) for image in images[start : start + self.batch_size]])
            batch = batch.to(self.device)
            with torch.inference_mode(), self._autocast():
                features = self._model.encode_image(batch, normalize=True)
            batches.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(batches, axis=0))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        torch = self._torch
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer(list(texts[start : start + self.batch_size])).to(self.device)
            with torch.inference_mode(), self._autocast():
                features = self._model.encode_text(tokens, normalize=True)
            batches.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(batches, axis=0))

    def _autocast(self):
        if self.device and self.device.startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return self._torch.autocast(device_type="cpu", enabled=False)

