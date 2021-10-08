import torch
import torch.nn as nn
from utils import sequential,Concat
from PartialConvolution import PartialConv2d
from torchsummary import summary

'''
def PBR(in_channels,out_channels,kernel_size,padding=1,stride=2):
    # partial convolution + batchnormalization + relu
    return sequential(PartialConv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,stride=stride,padding=padding),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True))

def PBL(in_channels,out_channels,kernel_size=3,padding=1,stride=1):
    # partial convolution + batchnormalization + leaky_relu
    return sequential(PartialConv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,stride=stride,padding=padding),
                      nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU(0.2))

def UpConcat(concat_rhs):
    # 反卷积加concat，resnet经典步骤
    return sequential(nn.Upsample(scale_factor=2,mode='nearest'),
                      Concat(concat_rhs))
'''

class PR(nn.Module): # 没有batchnorm
    def __init__(self,in_channels,out_channels,kernel_size,padding=1,stride=2,bias=False):
        super().__init__()
        self.pconv = PartialConv2d(in_channels,out_channels,kernel_size,
                                    padding=padding,stride=stride,bias=bias)
        self.relu = nn.ReLU(True)

    def forward(self,x,mask):
        x,updated_mask = self.pconv(x,mask)
        x = self.relu(x)
        return x,updated_mask

class PBR(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=1,stride=2,bias=False):
        super().__init__()
        self.pconv = PartialConv2d(in_channels,out_channels,kernel_size,
                                    padding=padding,stride=stride,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self,x,mask):
        x,updated_mask = self.pconv(x,mask)
        x = self.relu(self.bn(x))
        return x,updated_mask

class PBL(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False):
        super().__init__()
        self.pconv = PartialConv2d(in_channels,out_channels,kernel_size,
                                    padding=padding,stride=stride,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lk_relu = nn.LeakyReLU(0.2)

    def forward(self,x,mask):
        x,updated_mask = self.pconv(x,mask)
        x = self.lk_relu(self.bn(x))
        return x,updated_mask

class UpConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='nearest')
        self.concat = Concat()

    def forward(self,x,x_rhs,mask,mask_rhs):
        x = self.concat(self.up(x),x_rhs)
        mask = self.concat(self.up(mask),mask_rhs)
        return x,mask

class PConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [64,128,256,512,512,512,512,512]
        self.enc_kernels = [7,5,5,3,3,3,3,3]
        self.dec_kernels = 3
        self.enc_padding = [3,2,2,1,1,1,1,1] # 7*7对应padding=3,5*5的两个是2,3*3的是1
        self.PConv1 = PR(3,self.channels[0],self.enc_kernels[0],self.enc_padding[0])
        # 这个第一层独立出来，因为没有BN
        self.PConv2 = PBR(self.channels[0],self.channels[1],self.enc_kernels[1],self.enc_padding[1])
        self.PConv3 = PBR(self.channels[1],self.channels[2],self.enc_kernels[2],self.enc_padding[2])
        self.PConv4 = PBR(self.channels[2],self.channels[3],self.enc_kernels[3],self.enc_padding[3])
        self.PConv5 = PBR(self.channels[3],self.channels[4],self.enc_kernels[4],self.enc_padding[4])
        self.PConv6 = PBR(self.channels[4],self.channels[5],self.enc_kernels[5],self.enc_padding[5])
        self.PConv7 = PBR(self.channels[5],self.channels[6],self.enc_kernels[6],self.enc_padding[6])
        self.PConv8 = PBR(self.channels[6],self.channels[7],self.enc_kernels[7],self.enc_padding[7])
        # 这里是收缩

        # 下面是延伸，下面的padding应该全是1
        self.PConv9 = PBL(self.channels[7] + self.channels[6],self.channels[6]) # 512+512->512
        self.PConv10 = PBL(self.channels[6] + self.channels[5],self.channels[5]) # 512+512->512
        self.PConv11 = PBL(self.channels[5] + self.channels[4],self.channels[4]) # 512+512->512
        self.PConv12 = PBL(self.channels[4] + self.channels[3],self.channels[3]) # 512+512->512
        self.PConv13 = PBL(self.channels[3] + self.channels[2],self.channels[2]) # 512+256->256
        self.PConv14 = PBL(self.channels[2] + self.channels[1],self.channels[1]) # 256+128->128
        self.PConv15 = PBL(self.channels[1] + self.channels[0],self.channels[0]) # 128+64->64
        #self.PConv16 = PBL(self.channels[0] + 3,3,bias=True) # 64+3->3
        self.PConv16 = PartialConv2d(self.channels[0] + 3,3,3,padding=1,stride=1,bias=True)
        
    def forward(self,x,mask):
        # 假设输入是(1,3,512,512)
        enc_x1,enc_mask1 = self.PConv1(x,mask) # (1,64,256,256)
        enc_x2,enc_mask2 = self.PConv2(enc_x1,enc_mask1) # (1,128,128,128)
        enc_x3,enc_mask3 = self.PConv3(enc_x2,enc_mask2) # (1,256,64,64)
        enc_x4,enc_mask4 = self.PConv4(enc_x3,enc_mask3) # (1,512,32,32)
        enc_x5,enc_mask5 = self.PConv5(enc_x4,enc_mask4) # (1,512,16,16)
        enc_x6,enc_mask6 = self.PConv6(enc_x5,enc_mask5) # (1,512,8,8)
        enc_x7,enc_mask7 = self.PConv7(enc_x6,enc_mask6) # (1,512,4,4)
        enc_x8,enc_mask8 = self.PConv8(enc_x7,enc_mask7) # (1,512,2,2)

        dec_x8,dec_mask8 = UpConcat()(enc_x8,enc_x7,enc_mask8,enc_mask7)   # 512+512
        dec_x8,dec_mask8 = self.PConv9(dec_x8,dec_mask8)        # 512+512->512      

        dec_x7,dec_mask7 = UpConcat()(dec_x8,enc_x6,dec_mask8,enc_mask6)   # 512+512
        dec_x7,dec_mask7 = self.PConv10(dec_x7,dec_mask7)       # 512+512->512 

        dec_x6,dec_mask6 = UpConcat()(dec_x7,enc_x5,dec_mask7,enc_mask5)   # 512+512
        dec_x6,dec_mask6 = self.PConv11(dec_x6,dec_mask6)       # 512+512->512 

        dec_x5,dec_mask5 = UpConcat()(dec_x6,enc_x4,dec_mask6,enc_mask4)   # 512+512
        dec_x5,dec_mask5 = self.PConv12(dec_x5,dec_mask5)       # 512+512->512

        dec_x4,dec_mask4 = UpConcat()(dec_x5,enc_x3,dec_mask5,enc_mask3)   # 512+256
        dec_x4,dec_mask4 = self.PConv13(dec_x4,dec_mask4)       # 512+256->256

        dec_x3,dec_mask3 = UpConcat()(dec_x4,enc_x2,dec_mask4,enc_mask2)   # 256+128
        dec_x3,dec_mask3 = self.PConv14(dec_x3,dec_mask3)       # 256+128->128

        dec_x2,dec_mask2 = UpConcat()(dec_x3,enc_x1,dec_mask3,enc_mask1)   # 128+64
        dec_x2,dec_mask2 = self.PConv15(dec_x2,dec_mask2)       # 128+64->64

        dec_x1,dec_mask1 = UpConcat()(dec_x2,x,dec_mask2,mask)        # 64+3
        dec_x1,_ = self.PConv16(dec_x1,dec_mask1)       # 64+3->3
        return dec_x1

def main():
    model = PConvNet()
    model = model.cuda()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device) 
    x = torch.rand(size=(2,3,256,256)).cuda()
    mask = torch.rand(size=(2,3,256,256)).cuda()
    y = model(x,mask)
    #summary(model,(3,256,256))
    print(y.size())
    
if __name__ == '__main__':
    main()


