# smart-tool-select

Lightweight tool selector for binding relevant tools to LLMs per request.

## Install
```bash
pip install smart-tool-select
```
This package bundles the default embedding model and does not download at runtime.

## Quick usage
```python
from smart_tool_select import SmartToolSelector

filtered_tools = SmartToolSelector(
    "planing trip to dubai and hotel",
    tools,
    mode="semantic",
    top_k=3,
)
```

## Before vs After binding
```python
# BEFORE: bind all tools (e.g., 20 tools)
llm_with_tools = llm.bind_tools(tools)

# AFTER: bind only top_k semantic matches (e.g., 3 tools)
filtered_tools = SmartToolSelector(
    "planing trip to dubai and hotel",
    tools,
    mode="semantic",
    top_k=3,
)
llm_with_tools = llm.bind_tools(filtered_tools)
```

## Modes and accuracy
- `mode="lite"` (default): TF-IDF keyword matching. Fast and offline, but less semantic.
- `mode="semantic"`: Embedding-based matching. Higher accuracy for intent-heavy queries.

If you care about accuracy, use `mode="semantic"`. Use `mode="lite"` for maximum speed or offline environments.

## Optional extras
```bash
pip install smart-tool-select[faiss]
pip install smart-tool-select[hnswlib]
pip install smart-tool-select[yaml]
```
