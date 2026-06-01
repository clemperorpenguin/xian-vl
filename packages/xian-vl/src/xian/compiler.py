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

from __future__ import annotations

import os
import logging
from typing import Any

import yaml

from xian.url_safety import markdown_http_https_url_or_none

logger = logging.getLogger(__name__)

class WikiCompiler:
    """Generates Obsidian-compatible Markdown files with bidirectional linking.
    """
    
    def __init__(self, wiki_dir: str = "wiki"):
        self.wiki_dir = wiki_dir
        os.makedirs(self.wiki_dir, exist_ok=True)
        
    def compile(
        self,
        entity_name: str,
        content: str,
        metadata: dict[str, Any],
        infobox: dict[str, str] | None = None,
    ) -> str:
        """Compiles content into a Markdown file with YAML frontmatter.
        """
        # Ensure entity name is clean for filename
        safe_filename = "".join(c for c in entity_name if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_filename:
            safe_filename = "untitled"
        filename = f"{safe_filename}.md"
        filepath = os.path.join(self.wiki_dir, filename)
        
        # Defense-in-depth: ensure the resolved path is within the wiki directory
        from pathlib import Path
        try:
            Path(filepath).resolve().relative_to(Path(self.wiki_dir).resolve())
        except ValueError:
            raise ValueError(f"Entity name would escape wiki directory: {entity_name!r}")
        
        # Prepare frontmatter
        frontmatter = {
            "title": entity_name,
            "original_names": metadata.get("original_names", []),
            "sources": metadata.get("sources", []),
            "tags": ["lore", "auto-generated"],
        }
        
        # Add infobox to frontmatter or as a table
        if infobox:
            frontmatter["infobox"] = infobox
            
        yaml_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)
        
        # Build file content
        lines = [
            "---",
            yaml_str.strip(),
            "---",
            "",
            f"# {entity_name}",
            "",
        ]
        
        # Add infobox as a table at the top if present
        if infobox:
            lines.extend([
                "## Quick Facts",
                "",
                "| Property | Value |",
                "| --- | --- |",
            ])
            for key, val in infobox.items():
                lines.append(f"| {key} | {val} |")
            lines.append("")
            
        # Add main content
        lines.extend([
            "## Content",
            "",
            content,
            "",
            "## References",
            "",
        ])
        
        for source in metadata.get("sources", []):
            title = source.get("title", "Source")
            href = markdown_http_https_url_or_none(source.get("url"))
            if href:
                lines.append(f"- [{title}]({href})")
            else:
                lines.append(f"- {title} *(source URL omitted)*")
            
        # Enforce strict Obsidian-style bidirectional linking
        # This is a heuristic: if we see other known entities, we could link them.
        # For now, let's at least ensure the main entity is linkable by others,
        # and maybe link the original names if they look like entities.
        # The user mentioned "Enforce strict Obsidian-style bidirectional linking ([[Entity]])".
        # I'll add a section for related entities if provided.
        
        if "related_entities" in metadata:
            lines.extend([
                "",
                "## Related Entities",
                "",
            ])
            for related in metadata["related_entities"]:
                lines.append(f"- [[{related}]]")
                
        full_content = "\n".join(lines)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)
            
        logger.info("Wiki file written to %s", filepath)
        return filepath
