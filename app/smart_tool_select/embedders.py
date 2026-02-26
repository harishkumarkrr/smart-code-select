from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...


def _resolve_model_path(model_name: str, *, local_only: bool) -> str:
    if model_name == "all-MiniLM-L6-v2":
        local = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"
        if local.exists():
            return str(local)
        if local_only:
            raise RuntimeError(
                "Bundled model not found at "
                f"{local}. Run scripts/vendor_model.py before install."
            )
    return model_name


class SentenceTransformersEmbedder:
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", *, local_only: bool = False
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(
            _resolve_model_path(model_name, local_only=local_only)
        )

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=False)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self.model.encode(list(texts), normalize_embeddings=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


_DEFAULT_EMBEDDER: SentenceTransformersEmbedder | None = None


def get_default_embedder() -> SentenceTransformersEmbedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformersEmbedder(
            "all-MiniLM-L6-v2", local_only=True
        )
    return _DEFAULT_EMBEDDER
