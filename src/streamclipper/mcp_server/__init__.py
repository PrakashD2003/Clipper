"""MCP Server entry point using MCP SDK v2 (MCPServer) stdio transport."""
from .server import mcp


def main() -> None:
    """Run the StreamClipper MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
