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
     


     