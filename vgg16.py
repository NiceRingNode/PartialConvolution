import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union,List,Dict,Any,cast
from utils import weights_init

model_urls = {
    'vgg16':'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn':'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}

class VGG(nn.Module):
    def __init__(self,features:nn.Module,num_classes:int=1000,init_weights:bool=True) -> None:
        super().__init__()
        self.features = features # 这个只能按源代码这么叫，就是主干网络
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7,4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096,4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096,num_classes))
        if init_weights:
            weights_init()

    def forward(self,x):
        x = self.features(x) # 这个是主特征提取模型，分类器只是玩的
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def make_layers(cfg:List[Union[str,int]],batch_norm:bool=False) -> nn.Sequential:
    layers:List[nn.Module] = []
    in_channels = 3
    for v in cfg: # v是value，cfg是列表，cfgs是字典，值为列表
        if v == 'MaxPool':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            v = cast(int,v)
            conv = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv,nn.BatchNorm2d(v),nn.ReLU(True)]
            else:
                layers += [conv,nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)
    
cfgs:Dict[str,List[Union[str,int]]] = {
    '16':[64,64,'MaxPool',128,128,'MaxPool',256,256,256,'MaxPool',512,512,512,'MaxPool',
        512,512,512,'MaxPool']
    # vgg16是13层卷积，加3层maxpool，最后那个maxpool不算，输出512*7*7交给classifier
}

def vgg_impl(arch:str,cfg_key:str,batch_norm:bool,pretrained:bool,progress:bool,**kwargs:Any) -> VGG:
    # arch是vgg16_bn这种键，对应不同的网址
    # cfg_key是11,13,16,19分别代表不同的vgg层数
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg_key],batch_norm=batch_norm),**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained:bool=False,progress:bool=True,**kwargs:Any) -> VGG:
    # pretrained (bool): If True, returns a model pre-trained on ImageNet
    # progress (bool): If True, displays a progress bar of the download to stderr
    return vgg_impl('vgg16','16',False,pretrained,progress,**kwargs)

def vgg16_bn(pretrained:bool=False,progress:bool=True,**kwargs:Any) -> VGG:
    # pretrained (bool): If True, returns a model pre-trained on ImageNet
    # progress (bool): If True, displays a progress bar of the download to stderr
    return vgg_impl('vgg16_bn','16',True,pretrained,progress,**kwargs)