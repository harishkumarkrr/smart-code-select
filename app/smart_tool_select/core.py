from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .embedders import Embedder, get_default_embedder
from .indexes import InMemoryIndex, VectorIndex


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    raw: Any
    tags: Optional[List[str]] = None


@dataclass
class SelectionResult:
    tools: List[ToolDef]
    scores: List[Tuple[str, float]]
    explanation: Optional[Dict[str, Any]] = None


def _tool_text(tool: ToolDef) -> str:
    parts = [tool.name, tool.description or ""]
    if tool.tags:
        parts.append("Tags: " + ", ".join(tool.tags))
    return "\n".join(parts).strip()


def _normalize_vector(vec: Sequence[float]) -> List[float]:
    denom = (sum(x * x for x in vec) ** 0.5) + 1e-12
    return [x / denom for x in vec]


def _normalize_tool(tool: Any) -> ToolDef:
    if isinstance(tool, ToolDef):
        return tool

    if isinstance(tool, dict):
        name = tool.get("name")
        description = tool.get("description", "") or ""
        tags = tool.get("tags")
        if not name:
            raise ValueError("Tool dict missing 'name'")
        return ToolDef(name=name, description=description, raw=tool, tags=tags)

    name = getattr(tool, "name", None)
    if not name:
        raise ValueError("Tool object missing 'name'")
    description = getattr(tool, "description", "") or ""
    tags = getattr(tool, "tags", None)
    return ToolDef(name=name, description=description, raw=tool, tags=tags)


def freeze_tools(tools: Sequence[Any]) -> Tuple[Any, ...]:
    """
    Return an immutable tools tuple so the cache key remains stable.

    Use this when you load tools dynamically and want to ensure
    a single embedded index is reused across requests.
    """
    return tuple(tools)


class _ToolSelector:
    def __init__(
        self,
        tools: Sequence[Any],
        embedder: Optional[Embedder] = None,
        *,
        mode: str = "lite",
        index: Optional[VectorIndex] = None,
        top_k: int = 5,
        min_score: float = 0.2,
        always_include: Optional[Iterable[str]] = None,
        explain: bool = False,
    ) -> None:
        if mode not in {"lite", "semantic"}:
            raise ValueError("mode must be 'lite' or 'semantic'")
        self.tools = [_normalize_tool(t) for t in tools]
        self.mode = mode
        self.embedder = embedder or get_default_embedder()
        self.index = index or InMemoryIndex()
        self.top_k = top_k
        self.min_score = min_score
        self.always_include = set(always_include or [])
        self.explain = explain

        self._tool_matrix = None
        self._vectorizer = None
        self._build_index()

    def _build_index(self) -> None:
        texts = [_tool_text(t) for t in self.tools]
        if self.mode == "lite":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
            except Exception as exc:
                raise RuntimeError(
                    "scikit-learn is required for mode='lite'. "
                    "Install with: pip install scikit-learn"
                ) from exc
            self._vectorizer = TfidfVectorizer()
            self._tool_matrix = self._vectorizer.fit_transform(texts)
            return

        vectors = self.embedder.embed_batch(texts)
        vectors = [_normalize_vector(v) for v in vectors]
        self.index.add(vectors)

    def select(self, query: str) -> SelectionResult:
        if self.mode == "lite":
            if self._vectorizer is None or self._tool_matrix is None:
                raise RuntimeError("TF-IDF index not initialized")
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tool_matrix)[0]
            scored = [(tool, float(score)) for tool, score in zip(self.tools, scores)]
            scored.sort(key=lambda x: (-x[1], x[0].name))
        else:
            qvec = _normalize_vector(self.embedder.embed(query))
            ranked = self.index.search(qvec, top_k=max(self.top_k, 10))
            scored = [(self.tools[i], score) for i, score in ranked]
            scored.sort(key=lambda x: (-x[1], x[0].name))

        selected = [t for t, s in scored if s >= self.min_score][: self.top_k]
        for t in self.tools:
            if t.name in self.always_include and t not in selected:
                selected.append(t)

        explanation = None
        if self.explain:
            explanation = {
                "query": query,
                "top_k": self.top_k,
                "min_score": self.min_score,
                "always_include": sorted(self.always_include),
                "ranked": [(t.name, s) for t, s in scored[:10]],
                "selected": [(t.name, next(s for tt, s in scored if tt == t)) for t in selected],
            }

        return SelectionResult(
            tools=selected,
            scores=[(t.name, s) for t, s in scored],
            explanation=explanation,
        )

    def __call__(self, query: str) -> List[Any]:
        return [t.raw for t in self.select(query).tools]


_SELECTOR_CACHE: Dict[str, _ToolSelector] = {}


def SmartToolSelector(
    query: str,
    tools: Sequence[Any],
    **kwargs: Any,
) -> List[Any]:
    """
    Usage:
        filtered = SmartToolSelector(query, tools, top_k=5)
    """
    mode = kwargs.get("mode", "lite")
    embedder = kwargs.get("embedder")
    index = kwargs.get("index")
    top_k = kwargs.get("top_k")
    min_score = kwargs.get("min_score")
    always_include = tuple(sorted(kwargs.get("always_include") or []))
    explain = kwargs.get("explain")
    cache_key = (
        f"{id(tools)}:{mode}:{id(embedder)}:{type(index).__name__ if index else 'default'}:"
        f"{top_k}:{min_score}:{always_include}:{explain}"
    )
    selector = _SELECTOR_CACHE.get(cache_key)
    if selector is None:
        selector = _ToolSelector(tools, **kwargs)
        _SELECTOR_CACHE[cache_key] = selector
    return selector(query)
