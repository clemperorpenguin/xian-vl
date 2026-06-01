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

"""Tools for querying the JX3 knowledge base."""

from xian.knowledge.jx3_db import get_db

def query_jx3_database(query: str) -> str:
    """Search the JX3 database for a class or spec name and return detailed information.
    
    Args:
        query: The name of the class or spec to search for.
        
    Returns:
        A formatted string with the results or a "not found" message.
    """
    db = get_db()
    
    # Check if query matches a class
    cls = db.get_class_by_name(query)
    if cls:
        specs = db.get_specs_by_class(cls['id'])
        specs_str = "\n".join([f"  - {s['cn_name']} ({s['en_name']}): {s['role']} [{s['type']}]" for s in specs])
        return (f"Class: {cls['cn_name']} ({cls['en_name']})\n"
                f"Role: {cls['role']}\n"
                f"Description: {cls['description']}\n"
                f"Specs (XinFa):\n{specs_str}")
                
    # Check if query matches a spec
    specs = db.search_spec(query)
    if specs:
        results = []
        for s in specs:
            # Use the new get_class_by_id method to avoid O(n) loop
            c = db.get_class_by_id(s['school_id'])
            cls_name = f"{c['cn_name']} ({c['en_name']})" if c else "Unknown"
            
            results.append(f"Spec: {s['cn_name']} ({s['en_name']})\n"
                           f"Class: {cls_name}\n"
                           f"Role: {s['role']} [{s['type']}]")
        return "\n\n".join(results)
        
    return f"No results found in JX3 database for '{query}'."
