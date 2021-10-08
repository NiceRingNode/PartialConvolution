import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import PConvNet
from dataset import Places2
from PIL import Image
import argparse,os,time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid,save_image
from utils import denormalize

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--pretrained_root',type=str,
    default='./weights/checkpoint_mask_lightest_16.8.pth',
    help='the root of pretrained weights')
parser.add_argument('--dataset',type=str,default='mask_lightest',
    help='the mask dataset choose')
opt = parser.parse_args()

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
pconvnet = PConvNet().cuda()

img_mean = np.array([0.485,0.456,0.406])
img_std = np.array([0.229,0.224,0.225])
img_h = 256
img_w = 256

img_transform = transforms.Compose([
        transforms.Resize((img_h,img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean,std=img_std)
    ])
mask_transform = transforms.Compose([
        transforms.Resize((img_h,img_w)),
        transforms.ToTensor() # 归一化
    ])

val_dataset = Places2(train=False,mask_dataset=opt.dataset,
    img_transform=img_transform,mask_transform=mask_transform)

state_dict = torch.load(opt.pretrained_root) # './weights/checkpoint_mask_lightest_16.8.pth'
new_state_dict = {}
for k in state_dict.keys():
    new_k = k[7:]
    new_state_dict[new_k] = state_dict[k]
pconvnet.load_state_dict(new_state_dict)

pconvnet.eval()
# begin = np.random.randint(0,len(val_dataset) - opt.batch_size)
# mask_img,mask,y_true = zip(*[val_dataset[begin + i] for i in range(opt.batch_size)])
mask_img,mask,y_true = zip(*[val_dataset[i] for i in range(opt.batch_size)])
mask_img = torch.stack(mask_img).cuda()
mask = torch.stack(mask).cuda() # 这个是在最高维堆叠，(3,256,256)变成(6,3,256,256)
y_true = torch.stack(y_true).cuda()
start = time.time()
with torch.no_grad():
    y_pred = pconvnet(mask_img,mask)
print(f'time elapsed: {((time.time() - start) * 1000.):5.2f}ms')
y_pred = y_pred.cuda()
y_comp = mask * mask_img + (1 - mask) * y_pred
#print(mask[0].cpu())
img_grid = make_grid(
        torch.cat((denormalize(mask_img,torch.device('cuda')),
            denormalize(y_true,torch.device('cuda')),denormalize(y_pred,torch.device('cuda')),
    denormalize(y_comp,torch.device('cuda')),mask),dim=0))
# img_grid = make_grid(torch.cat((mask_img,y_true,y_pred,y_comp,mask),dim=0))
save_image(img_grid,'./output/result.png')

# y_pred = transforms.ToPILImage()(denormalize(y_pred,torch.device('cuda')).view(-1,256,256))
# #y_pred = transforms.ToPILImage()(y_pred.view(-1,256,256))
# #y_pred.show()
# #y_comp = transforms.ToPILImage()(y_comp.view(-1,256,256))
# y_comp = transforms.ToPILImage()(denormalize(y_comp,torch.device('cuda')).view(-1,256,256))
# #y_comp.show()
# mask = transforms.ToPILImage()(denormalize(mask,torch.device('cuda')).view(-1,256,256))
# #mask.show()