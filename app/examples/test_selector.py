import asyncio

from fastmcp.client import Client
from smart_tool_select import SmartToolSelector, freeze_tools


async def main() -> None:
    client = Client("http://127.0.0.1:8001/mcp")
    async with client:
        tools = await client.list_tools()

    tools = freeze_tools(tools)

    query = "book a flight to New York"
    filtered = SmartToolSelector(query, tools, top_k=5)

    print("Query:", query)
    print("Selected tools:", [t.name for t in filtered])


if __name__ == "__main__":
    asyncio.run(main())
