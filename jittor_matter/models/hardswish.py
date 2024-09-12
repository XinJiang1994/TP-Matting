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

from jittor import nn
import jittor as jt

class Hardsigmoid(nn.Module):
     def __init__(self, *args, **kw) -> None:
          super().__init__(*args, **kw)
          self.relu6=jt.nn.ReLU6()

     def execute(self, x) -> None:
          return self.relu6(x+3)/6

        

class Hardswish(nn.Module):
     def __init__(self, *args, **kw) -> None:
          super().__init__(*args, **kw)
          self.hard_sigmoid= Hardsigmoid()

     def execute(self, x) -> None:
          return x * self.hard_sigmoid(x)
     


     