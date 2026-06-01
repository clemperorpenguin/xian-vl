# LORE — Interactive research Wiki builder CLI.
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

import asyncio
from lore.scraper import PlaywrightScraper

async def main():
    s = PlaywrightScraper()
    res = await s.scrape("https://www.jx3box.com/community/6220")
    print("HTML Content length:", len(res.get("html", "")))
    print("HTML Content snippet:", res.get("html", "")[:500])

if __name__ == "__main__":
    asyncio.run(main())
