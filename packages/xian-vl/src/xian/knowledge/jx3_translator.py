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

"""Translator for JX3 game terms using the database."""

from xian.knowledge.jx3_db import get_db

def translate_to_en(cn_name: str) -> str:
    """Translate a JX3 class or spec name from Chinese to English.
    
    Args:
        cn_name: The Chinese name to translate.
        
    Returns:
        The English name if found, otherwise the original Chinese name.
    """
    db = get_db()
    
    # Try classes
    cls = db.get_class_by_name(cn_name)
    if cls and cls['cn_name'] == cn_name:
        return cls['en_name']
        
    # Try specs
    specs = db.search_spec(cn_name)
    for spec in specs:
        if spec['cn_name'] == cn_name:
            return spec['en_name']
            
    return cn_name

def translate_to_cn(en_name: str) -> str:
    """Translate a JX3 class or spec name from English to Chinese.
    
    Args:
        en_name: The English name to translate.
        
    Returns:
        The Chinese name if found, otherwise the original English name.
    """
    db = get_db()
    
    # Try classes
    cls = db.get_class_by_name(en_name)
    if cls and cls['en_name'].lower() == en_name.lower():
        return cls['cn_name']
        
    # Try specs
    specs = db.search_spec(en_name)
    for spec in specs:
        if spec['en_name'].lower() == en_name.lower():
            return spec['cn_name']
            
    return en_name
