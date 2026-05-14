from __future__ import annotations

import os
import logging
from typing import Any

import yaml

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
            lines.append(f"- [{source.get('title', 'Source')}]({source.get('url')})")
            
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
