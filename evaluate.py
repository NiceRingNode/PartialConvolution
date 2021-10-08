import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Places2
from torchvision import transforms
import numpy as np
from torchvision.utils import make_grid,save_image
import argparse
from loss import loss
from utils import denormalize

class Opt:
    def __init__(self):
        self.img_h = 256
        self.img_w = 256
        self.batch_size = 8

opt = Opt()

img_mean = np.array([0.485,0.456,0.406])
img_std = np.array([0.229,0.224,0.225])

img_transform = transforms.Compose([
        transforms.Resize((opt.img_h,opt.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean,std=img_std)
    ])
mask_transform = transforms.Compose([
        transforms.Resize((opt.img_h,opt.img_w)),
        transforms.ToTensor() # 归一化
    ])

val_dataset = Places2(train=False,mask_dataset='mask_light',
    img_transform=img_transform,mask_transform=mask_transform)

def validate(model,filename):
    model.eval()
    begin = np.random.randint(0,len(val_dataset) - opt.batch_size)
    mask_img,mask,y_true = zip(*[val_dataset[begin + i] for i in range(opt.batch_size)])
    mask_img = torch.stack(mask_img)
    mask = torch.stack(mask) # 这个是在最高维堆叠，(3,256,256)变成(6,3,256,256)
    y_true = torch.stack(y_true)
    with torch.no_grad():
        y_pred = model(mask_img,mask)
    mask_img = mask_img.cuda()
    mask = mask.cuda()
    y_true = y_true.cuda()
    y_pred = y_pred.cuda()
    cur_loss = loss(mask_img,y_true,y_pred,mask)
    y_comp = mask * mask_img + (1 - mask) * y_pred
    img_grid = make_grid(
            torch.cat((denormalize(mask_img,torch.device('cuda')),
                denormalize(y_true,torch.device('cuda')),
                denormalize(y_pred,torch.device('cuda')),
        denormalize(y_comp,torch.device('cuda')),mask),dim=0))
    # img_grid = make_grid(torch.cat((mask_img,y_true,y_pred,y_comp,mask),dim=0))
    save_image(img_grid,filename)
    return cur_loss


            