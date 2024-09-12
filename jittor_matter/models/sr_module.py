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


import jittor as jt
import jittor.nn as nn

class SRNet(nn.Module):
    def __init__(self,num_tokens):
        super().__init__()
        self.conv1x = nn.Sequential(
            nn.Conv2d(num_tokens+3,16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        self.conv_out=nn.Conv2d(16,1,kernel_size=3,stride=1,padding=1)

    def execute(self,src,lr):
        x= self.conv1x(jt.cat((src,lr),dim=1))
        x=self.conv_out(x)
        return x


