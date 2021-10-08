import torch
import torch.nn as nn
from vgg16 import vgg16,vgg16_bn

def pixel_loss(y_true,y_pred,mask):
    '''
        y_true是gt图像,(c,h,w)
        y_pred是预测的,(c,h,w)
        mask是给图像用的binary_mask,0是洞,(h,w)
        这个loss包含原文提到的Lhole,Lvalid,是L1 loss
    '''
    _,c,h,w = y_true.size()
    loss_fn = nn.L1Loss(reduction='mean')
    # Ngt = c * h * w # 不用，自动求的
    hole_loss = loss_fn((1 - mask) * y_pred,(1 - mask) * y_true)
    valid_loss = loss_fn(mask * y_pred,mask * y_true)
    return hole_loss,valid_loss

class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485,0.456,0.406]).view(-1,1,1).cuda()
        self.std = torch.tensor([0.229,0.224,0.225]).view(-1,1,1).cuda() # 广播

    def forward(self,x):
        # 如果这里的输入还没有归一化的，就要在数据预处理的时候先归一化
        return torch.div(x - self.mean,self.std)

VGG16 = vgg16(pretrained=True).cuda() # 全局的不用每次都重复调用
extract_1 = nn.Sequential(*VGG16.features[:5])
extract_2 = nn.Sequential(*VGG16.features[5:10])
extract_3 = nn.Sequential(*VGG16.features[10:17])
for p in extract_1.parameters():
    p.requires_grad = False
for p in extract_2.parameters():
    p.requires_grad = False
for p in extract_3.parameters():
    p.requires_grad = False

def vgg16extract(x): # x指的是y_true、y_pred或y_comp
    extract_res = []
    extract_res.append(extract_1(x))
    extract_res.append(extract_2(extract_res[-1]))
    extract_res.append(extract_3(extract_res[-1]))
    return extract_res

def gram_matrix(x): # x指的是y_true、y_pred或y_comp
    n,c,h,w = x.size()
    x = x.view(n,c,h * w)
    x_t = x.permute(0,2,1)
    gram = torch.div(torch.bmm(x,x_t),(c * h * w)) # 论文里面有这个因子
    return gram

def perceptual_style_tv_loss(mask_img,y_true,y_pred,mask):
    '''
        mask还是一样的mask，因为求loss的地方是输入输出都在的，所以comp可以计算出来
        perceptual loss:with the non-hole pixels directly set to ground truth;
            也就是用pred的非孔像素直接设置给原来gt的变白了的像素（孔像素）
        style loss:和perceptual loss一样要用到vgg16提取的高级特征，所以一起来搞
        可以用个类，然后只实例化一次反复调用；也可以设置个全局的，方便函数调用不用重复生成

        vgg16 extractor是抽取三个位置，[0,5),[5,10),[10,17),然后分别输入y_pred、y_comp和gt,
        各自得到三个feature map（pred三个，comp三个，gt三个），然后将pred和comp的feature map
        都和gt对应的位置作loss，但是原论文是每一层都做一次loss，看看哪个好

        tv loss要用y_comp，直接在这里搞好算了
    '''
    y_comp = mask * mask_img + (1 - mask) * y_pred
    loss_fn = nn.L1Loss(reduction='mean') # 默认是mean的
    
    features_pred = vgg16extract(y_pred)
    features_true = vgg16extract(y_true)
    features_comp = vgg16extract(y_comp)
    
    perceptual_loss = 0.
    style_loss = 0.
    for i in range(len(features_pred)):
        perceptual_loss += loss_fn(features_true[i],features_pred[i])
        perceptual_loss += loss_fn(features_true[i],features_comp[i])
        style_loss += loss_fn(gram_matrix(features_true[i]),gram_matrix(features_pred[i]))
        style_loss += loss_fn(gram_matrix(features_true[i]),gram_matrix(features_comp[i]))

    tv_loss = total_variation_loss(y_comp)

    return perceptual_loss,style_loss,tv_loss

def total_variation_loss(y_comp):
    return torch.mean(torch.abs(y_comp[:,:,:,:-1] - y_comp[:,:,:,1:])) + \
           torch.mean(torch.abs(y_comp[:,:,:-1,:] - y_comp[:,:,1:,:]))

# 各个loss的权重
lambda_valid = 1
lambda_hole = 6
lambda_perceptual = 0.05
lambda_style = 120
lambda_tv = 0.1

def loss(mask_img,y_true,y_pred,mask):
    hole_loss,valid_loss = pixel_loss(y_true,y_pred,mask)
    perceptual_loss,style_loss,tv_loss = perceptual_style_tv_loss(mask_img,y_true,y_pred,mask)
    total_loss = lambda_valid * valid_loss + lambda_hole * hole_loss + \
        lambda_perceptual * perceptual_loss + lambda_style * style_loss + lambda_tv * tv_loss
    return total_loss