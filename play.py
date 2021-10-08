import torch
import torch.nn as nn

class PBL(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=2):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,padding=padding,stride=stride),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.2))
    def forward(self,x):
        x = self.layers(x)
        return x

a = PBL(2,3,3,1,2)

class b(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = nn.Linear(3,64)
        self.k2 = nn.Linear(64,3)

    def forward(self,x):
        x = self.k1(x)
        x = self.k2(x)
        return x

# x = torch.rand((1,3)).cuda()
# btest = b()
# btest = btest.cuda()
# print(btest(x))

# a = torch.randint(0,2,(2,10))
# print(1 - a)

print(1 / (2 * 3.14 * 3.6 * 10 ** 7))
a = 1 / (2 * 3.14 * 3.6 * 10 ** 7)
c = (8 * 10 ** (-11)) ** 0.5
print(c)
l = a / c
print(l ** 2 * 10 ** 9)
print(1 / (2 * 3.14 * (150 * 10 ** (-9) * 68 * 10 ** (-12)) ** 0.5) / 10**7)