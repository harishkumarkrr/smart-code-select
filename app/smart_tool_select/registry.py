from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core import ToolDef, _normalize_tool


def _substitute_env(raw: str) -> str:
    for key, value in os.environ.items():
        raw = raw.replace("${" + key + "}", value)
    return raw


def load_registry(path: str, *, env_substitute: bool = True) -> Dict[str, Any]:
    p = Path(path)
    raw = p.read_text()
    if env_substitute:
        raw = _substitute_env(raw)

    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pyyaml is required for YAML registries. "
                "Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Registry file must be a JSON/YAML object")
    return data


async def _fetch_tools_for_server(
    name: str,
    cfg: Dict[str, Any],
    *,
    timeout_seconds: Optional[float] = None,
) -> Tuple[str, List[Any]]:
    try:
        from fastmcp.client import Client
    except Exception as exc:
        raise RuntimeError(
            "fastmcp is required to load tools from MCP servers."
        ) from exc

    client = Client(cfg)
    session_kwargs: Dict[str, Any] = {}
    if timeout_seconds is not None:
        session_kwargs["read_timeout_seconds"] = timeout_seconds

    async with client:
        tools = await client.list_tools()
    return name, tools


def _prefix_tool_name(tool: Any, prefix: str, sep: str) -> Any:
    new_name = f"{prefix}{sep}{tool.name}"
    if isinstance(tool, dict):
        tool = dict(tool)
        tool["name"] = new_name
        return tool
    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"name": new_name})
    if hasattr(tool, "copy"):
        copied = tool.copy()
        if hasattr(copied, "name"):
            copied.name = new_name
        return copied
    if hasattr(tool, "name"):
        tool.name = new_name
    return tool


async def load_tools_from_registry(
    registry_path: str,
    *,
    prefix_tools: bool = False,
    prefix_sep: str = "_",
    include_servers: Optional[Iterable[str]] = None,
    exclude_servers: Optional[Iterable[str]] = None,
    timeout_seconds: Optional[float] = None,
) -> List[ToolDef]:
    registry = load_registry(registry_path)
    servers = registry.get("mcpServers") or registry.get("servers")
    if not isinstance(servers, dict):
        raise ValueError("Registry must contain 'mcpServers' or 'servers' map.")

    include = set(include_servers or [])
    exclude = set(exclude_servers or [])

    tasks = []
    for name, cfg in servers.items():
        if include and name not in include:
            continue
        if name in exclude:
            continue
        if not isinstance(cfg, dict):
            continue
        tasks.append(_fetch_tools_for_server(name, cfg, timeout_seconds=timeout_seconds))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_defs: List[ToolDef] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        name, tools = result
        for tool in tools:
            if prefix_tools:
                tool = _prefix_tool_name(tool, prefix=name, sep=prefix_sep)
            tool_def = _normalize_tool(tool)
            tool_defs.append(tool_def)

    return tool_defs


def load_tools_from_registry_sync(
    registry_path: str,
    **kwargs: Any,
) -> List[ToolDef]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use load_tools_from_registry in async contexts.")
    except RuntimeError:
        loop = None

    return asyncio.run(load_tools_from_registry(registry_path, **kwargs))
