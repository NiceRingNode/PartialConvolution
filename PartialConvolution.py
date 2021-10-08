import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import weights_init

class PartialConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
        #super().__init__(in_channels,out_channels,kernel_size,stride,padding,bias=bias)
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=padding,bias=bias)
        # self.mask_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,
        #     padding=padding,bias=False)
        self.conv.apply(weights_init('kaiming'))

        self.weight_mask = torch.ones((out_channels,in_channels,kernel_size,kernel_size)).cuda()
        self.sum1 = in_channels * (kernel_size ** 2) # 原论文的sum1

    def forward(self,x,mask):
        # x:(n,c,h,w)
        # _,_,h,w = x.size()
        # mask = torch.randint(0,2,(h,w)) # 这就是01矩阵的mask
        # ones_cnt = sum(mask == 1)
        # factor = torch.FloatTensor(ones_cnt / (h * w))
        # x *= mask
        # x *= factor
        '''
            第一步，将mask和原来的x乘起来进行一次卷积，得到类似raw_output的东西
            第二步，将mask进行一次专属的卷积，然后得到一个update_mask，这是从源代码里看来的，
                原文论文里根本没有这一步
            第三步，计算ratio = sum1 / sumM
            第四步，将ratio乘上去，因为论文里面的b是乘了ratio再加的，所以要先把b减掉，
                乘了再将b加回去；按理来说论文到这里就完了但是，不知道为什么还要再乘个update_mask
        '''
        raw_y = self.conv(torch.mul(x,mask))

        with torch.no_grad():
            #updated_mask = self.mask_conv(mask)
            updated_mask = F.conv2d(mask,self.weight_mask,stride=self.stride,padding=self.padding)
            ratio = self.sum1 / (updated_mask + 1e-8) # sum1 / sumM
            updated_mask = torch.clamp(updated_mask,min=0,max=1)
        # stride收缩的时候是2，扩展的时候是1，padding也会变，所以这两个要指定

        if self.conv.bias is not None: # bias本身的shape是(out_channels,),所以要reshape
            bias_reshape = self.conv.bias.view(1,self.out_channels,1,1)
            y = torch.mul(raw_y - bias_reshape,ratio) + bias_reshape
            y = torch.mul(y,updated_mask)
        else:
            y = torch.mul(raw_y,ratio)
        return y,updated_mask

def main():
    x = np.random.randn(1,1,7,7)
    x = torch.FloatTensor(x).cuda()
    mask = torch.randn(1,1,7,7).cuda()
    model = PartialConv2d(1,3,3,1,1).cuda()
    x,up_mask = model(x,mask)
    #print(x)
    print(x.size())
    print(mask.equal(up_mask))

if __name__ == '__main__':
    main()

