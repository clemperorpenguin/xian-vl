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

import os
import shutil
import tempfile
import pytest
from xian.pipeline import VLProcessor, VLConfig
from xian.context_manager import FrameContext

@pytest.fixture
def temp_wiki_dir():
    # Create a temporary directory for wiki files
    tmpdir = tempfile.mkdtemp()
    
    # Write Li_Xiaoyao.md with list of original names
    with open(os.path.join(tmpdir, "Li_Xiaoyao.md"), "w", encoding="utf-8") as f:
        f.write("---\n"
                "title: Li Xiaoyao\n"
                "original_names: [李逍遥]\n"
                "---\n"
                "# Li Xiaoyao\n"
                "A famous character.")

    # Write Sect.md with list of original names
    with open(os.path.join(tmpdir, "Sect.md"), "w", encoding="utf-8") as f:
        f.write("---\n"
                "title: Sect\n"
                "original_names:\n"
                "  - 门派\n"
                "  - 宗门\n"
                "---\n"
                "# Sect\n"
                "A cultivation organization.")

    # Write invalid.md with broken YAML to test resilience
    with open(os.path.join(tmpdir, "invalid.md"), "w", encoding="utf-8") as f:
        f.write("---\n"
                "title: Broken\n"
                "original_names: [:\n"
                "---\n"
                "Content")

    yield tmpdir
    
    # Cleanup
    shutil.rmtree(tmpdir)

def test_load_glossary_from_wiki(temp_wiki_dir):
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = temp_wiki_dir
    
    glossary = processor.load_glossary_from_wiki()
    assert glossary["李逍遥"] == "Li Xiaoyao"
    assert glossary["门派"] == "Sect"
    assert glossary["宗门"] == "Sect"
    # Ensure invalid one was safely skipped
    assert "Broken" not in glossary.values()

def test_create_prompt_glossary_injection(temp_wiki_dir):
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = temp_wiki_dir
    
    system_prompt, user_prompt = processor.create_prompt(
        source_lang="Chinese",
        target_lang="English",
        mode="Game",
        styles=[]
    )
    
    assert "GLOSSARY" in system_prompt or "glossary" in system_prompt
    assert "李逍遥" in system_prompt
    assert "Li Xiaoyao" in system_prompt
    assert "门派" in system_prompt
    assert "Sect" in system_prompt

def test_create_prompt_rag_injection(temp_wiki_dir):
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = temp_wiki_dir
    
    # Mock context manager frame to simulate recently translated text
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    processor.context_manager.add_frame(img)
    processor.context_manager.update_last_frame_data(
        extracted_text="ORIGINAL:\n我的李逍遥在这里。\n\nTRANSLATED:\nMy Li Xiaoyao is here.",
        translations=[]
    )
    
    recent_text = processor.get_recent_text_for_search()
    assert recent_text == "我的李逍遥在这里。"
    
    system_prompt, _ = processor.create_prompt(
        source_lang="Chinese",
        target_lang="English",
        mode="Game",
        styles=[]
    )
    
    assert "LORE REFERENCE ARTICLES:" in system_prompt
    assert "Li Xiaoyao" in system_prompt
    assert "A famous character" in system_prompt

def test_create_cinematic_prompt_rag_injection(temp_wiki_dir):
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = temp_wiki_dir
    
    system_prompt = processor.create_cinematic_prompt(
        transcript="这里有一个神秘的门派",
        target_lang="English",
        styles=[]
    )
    
    assert "GLOSSARY" in system_prompt or "glossary" in system_prompt
    assert "LORE REFERENCE ARTICLES:" in system_prompt
    assert "Sect" in system_prompt
    assert "A cultivation organization" in system_prompt
