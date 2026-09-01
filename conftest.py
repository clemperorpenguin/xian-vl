# Xian-VL — Real-Time Vision-Language Assistant for Gaming Environments.
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

"""Workspace-wide pytest configuration.

The benchmark corpus itself is resolved by :mod:`benchmark_corpus`, which the
benchmark files import directly — pytest chooses its rootdir from the arguments
it is given, so this file is not loaded when someone runs a single test file
out of a workspace package.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark: measures the live pipeline against the real screenshot corpus",
    )
