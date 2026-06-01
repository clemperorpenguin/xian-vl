#!/usr/bin/env python3
import os
import re

python_template = """# {program_description}
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

java_kotlin_template = """/*
 * {program_description}
 * Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)
 */
"""

descriptions = [
    ("packages/xian-vl", "Xian-VL — Core Vision-Language orchestration engine."),
    ("packages/shared-types", "Xian-VL Shared Types — Canonical model definitions and constants."),
    ("apps/mage-client", "MAGE — Gaming HUD for real-time screen translation."),
    ("apps/lore-client", "LORE — Interactive research Wiki builder CLI."),
    ("apps/luduan-client", "Luduan — EPUB to RoboBook narration CLI."),
    ("apps/nate", "MAGE Companion — Android OCR and local dictionary client."),
    ("scripts", "Xian-VL Scripts — Development and automation scripts."),
]

def get_description(rel_path):
    for prefix, desc in descriptions:
        if rel_path.startswith(prefix):
            return desc
    return "Xian-VL — Real-Time Vision-Language Assistant for Gaming Environments."

def process_file(file_path, rel_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ('.py', '.kt', '.java'):
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "This program is free software" in content:
        return
        
    desc = get_description(rel_path)
    
    if ext == '.py':
        header = python_template.format(program_description=desc)
        shebang_match = re.match(r'^(#!.*?)\n', content)
        if shebang_match:
            shebang = shebang_match.group(1)
            rest = content[len(shebang) + 1:]
            new_content = f"{shebang}\n{header}\n{rest}"
        else:
            new_content = f"{header}\n{content}"
    else:
        header = java_kotlin_template.format(program_description=desc)
        new_content = f"{header}\n{content}"
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Added GPL v3 header to {rel_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exclude_dirs = {'.venv', '.git', '.github', '.pytest_cache', 'build', 'dist', 'lemonade'}
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, base_dir)
            process_file(file_path, rel_path)

if __name__ == "__main__":
    main()
