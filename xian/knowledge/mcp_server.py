"""
MCP (Model Context Protocol) Server for JX3 Knowledge Base.
This allows Cursor, Lemonade, or any MCP-compatible client to query the database.

Requires: pip install mcp
"""
import sys
import logging
from mcp.server.fastmcp import FastMCP
from xian.knowledge.tools import query_jx3_database

logging.basicConfig(level=logging.INFO)

# Create an MCP server instance
mcp = FastMCP("JX3 Knowledge Server")

@mcp.tool()
def search_jx3(query: str) -> str:
    """
    Search the JX3 Online (剑网3) game database for information about classes, specs, and game vocabulary.
    Use this to look up translations, roles, or class properties when the user asks about JX3.
    """
    return query_jx3_database(query)

if __name__ == "__main__":
    # Run the server on stdio (standard for MCP)
    mcp.run()
