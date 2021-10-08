import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import glob,argparse
from generate_mask import generate_mask

class Places2(Dataset):
    def __init__(self,train=True,mask_dataset='mask',img_transform=None,mask_transform=None):
        super().__init__()
        self.root_dir = r'./data/places365_standard/'
        if mask_dataset == 'mask':
            self.mask_dir = r'./data/mask/'
        elif mask_dataset == 'mask_light':
            self.mask_dir = r'./data/mask_light/'
        elif mask_dataset == 'mask_lightest':
            self.mask_dir = r'./data/mask_lightest/'
        self.imgs_dir = self.root_dir + 'train.txt' if train else self.root_dir + 'val.txt'
        self.img_height,self.img_width = (256,256)
        # 这两个文件里面有路径，其实就跟voc一样

        with open(self.imgs_dir,'r',encoding='utf-8') as f:
            self.imgs_path = f.readlines()
            self.imgs_path = [self.root_dir + i.strip() for i in self.imgs_path]
            #self.imgs_path = self.imgs_path[:100]
        self.masks_path = glob.glob(self.mask_dir + '*.png')

        self.imgs_cnt = len(self.imgs_path)
        self.masks_cnt = len(self.masks_path)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return self.imgs_cnt

    def __getitem__(self,index):
        img = Image.open(self.imgs_path[index % self.imgs_cnt]).convert('RGB') 
        # 模只是保险，其实不会超的
        mask = Image.open(self.masks_path[np.random.randint(0,self.masks_cnt)]).convert('RGB')
        #mask = Image.open('./07772.png').convert('RGB')
        # mask = np.asarray(mask).copy()
        # 随便选一个，就只能在这些里面选
        # mask = np.expand_dims(mask,axis=-1)
        # anti_mask_img = img * (1 - mask)
        # anti_mask_img = Image.fromarray(anti_mask_img)
        # zero_mask = np.stack((np.reshape((mask == 0),(self.img_height,self.img_width)),) * 3,axis=-1)
        # mask_img[zero_mask] += 255
        # mask[mask == 0] += 255 # 本来是1和255，ToTensor会归一化，所以变成了0到1之间
        # mask = Image.fromarray(mask).convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = np.expand_dims(mask,axis=-1)
        mask_img = img * mask
        return mask_img,mask,img # img要返回的，因为要给loss

def main():
    # with open('./data/places365_standard/train.txt','r',encoding='utf-8') as f:
    #     images_name = f.readlines()
    #     images_name = [i.strip() for i in images_name]
    # print(images_name[:10])

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--img_h',type=int,default=256)
    parser.add_argument('--img_w',type=int,default=256)
    opt = parser.parse_args()

    img_mean = np.array([0.485,0.456,0.406])
    img_std = np.array([0.229,0.224,0.225])

    img_transform = transforms.Compose([
            transforms.Resize((opt.img_h,opt.img_w)),
            transforms.ToTensor(),
            
        ])
    mask_transform = transforms.Compose([
            transforms.Resize((opt.img_h,opt.img_w)),
            transforms.ToTensor()
        ])
    dataset = Places2(img_transform=img_transform,mask_transform=mask_transform)
    num = 0
    for i in dataset:
        # mask_img,mask,img = i
        # mask_img = transforms.ToPILImage()(mask_img)
        # mask_img.show()
        # #print(mask)
        # mask = transforms.ToPILImage()(mask)
        # mask.show()
        if num == 0:
            break
        num += 1

if __name__ == '__main__':
    main()