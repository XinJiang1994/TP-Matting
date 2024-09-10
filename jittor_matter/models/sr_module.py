

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


