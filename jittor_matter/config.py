# Copyright (C) 2024 Jiang Xin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import yaml
from pathlib import Path

import os


def load_config():
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r"), Loader=yaml.FullLoader
    )


def check_os_environ(key, use):
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required for {use}."
        )


def dataset_dir():
    check_os_environ("DATASET", "data loading")
    return os.environ["DATASET"]
