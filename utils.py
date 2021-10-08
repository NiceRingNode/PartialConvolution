import torch
import torch.nn as nn
from functools import reduce

def sequential(*layers):
    if layers:
        return reduce(lambda f,g:lambda *args,**kwargs:g(f(*args,**kwargs)),layers)
    else:
        raise ValueError('composition of empty layer sequence is not supported')

class Concat(nn.Module):
    # nchw,rhs是被concat的那个
    def __init__(self):
        super().__init__()

    def forward(self,x,rhs): # x是concat的主体
        x = torch.cat((x,rhs),dim=1)
        return x

def weights_init(weights_type='gaussian'):
    def init_impl(m):
        classname = m.__class__.__name__
        if classname.find('Conv') == 0 and hasattr(m,'weight'):
            # 返回的是找到该子串的初始位置
            if weights_type == 'gaussian':
                nn.init.normal_(m.weight,0.0,0.02)
            elif weights_type == 'xavier':
                nn.init.xavier_normal_(m.weight,gain=2 ** 0.5)
            elif weights_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
            elif weights_type == 'orthogonal':
                nn.init.orthogonal_(m.weight,gain=2 ** 0.5)
            elif weights_type == 'default':
                pass
            else:
                assert 0,f'unspported initialzed mode: {weights_type}'
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.constant_(m.bias,0.0) # 不要偏置
        elif classname.find('BatchNorm') == 0 and hasattr(m,'weight'):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif classname.find('Linear') == 0 and hasattr(m,'weight'):
            nn.init.constant_(m.weight,0.01)
            nn.init.constant_(m.bias,0)

    return init_impl

img_mean = (0.485,0.456,0.406)
img_std = (0.229,0.224,0.225)

def denormalize(x,device):
    x = x.transpose(1,3)
    x = x * torch.tensor(img_std).to(device) + torch.tensor(img_std).to(device)
    x = x.transpose(1,3)
    return x