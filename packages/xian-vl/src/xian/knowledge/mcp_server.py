# Xian-VL — Core Vision-Language orchestration engine.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""
MCP (Model Context Protocol) Server for JX3 Knowledge Base.
This allows Cursor, Lemonade, or any MCP-compatible client to query the database.

Requires: pip install xian-vl[mcp]
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
