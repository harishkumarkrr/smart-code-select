from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple


class VectorIndex(Protocol):
    def add(self, vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[int, float]]: ...


class InMemoryIndex:
    def __init__(self) -> None:
        self.vectors: List[List[float]] = []

    def add(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors.extend([list(v) for v in vectors])

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        scores = []
        for i, v in enumerate(self.vectors):
            score = sum(a * b for a, b in zip(query_vec, v))
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class FaissIndex:
    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "faiss is required. Install with: pip install faiss-cpu"
            ) from exc

        self.faiss = faiss
        self.index = None
        self.dim = None

    def add(self, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        import numpy as np  # type: ignore

        mat = np.array(vectors, dtype="float32")
        if self.index is None:
            self.dim = mat.shape[1]
            self.index = self.faiss.IndexFlatIP(self.dim)
        self.index.add(mat)

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        import numpy as np  # type: ignore

        q = np.array([query_vec], dtype="float32")
        scores, indices = self.index.search(q, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


class HnswlibIndex:
    def __init__(self, *, ef_construction: int = 200, ef_search: int = 64, M: int = 16) -> None:
        try:
            import hnswlib  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "hnswlib is required. Install with: pip install hnswlib"
            ) from exc

        self.hnswlib = hnswlib
        self.index = None
        self.dim = None
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self._count = 0

    def add(self, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        import numpy as np  # type: ignore

        mat = np.array(vectors, dtype="float32")
        if self.index is None:
            self.dim = mat.shape[1]
            self.index = self.hnswlib.Index(space="cosine", dim=self.dim)
            self.index.init_index(
                max_elements=len(vectors),
                ef_construction=self.ef_construction,
                M=self.M,
                random_seed=42,
            )
        else:
            self.index.resize_index(self._count + len(vectors))

        labels = list(range(self._count, self._count + len(vectors)))
        self.index.add_items(mat, labels)
        self.index.set_ef(self.ef_search)
        self._count += len(vectors)

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        import numpy as np  # type: ignore

        q = np.array([query_vec], dtype="float32")
        labels, distances = self.index.knn_query(q, k=top_k)
        results: List[Tuple[int, float]] = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx < 0:
                continue
            score = 1.0 - float(dist)
            results.append((int(idx), score))
        return results
