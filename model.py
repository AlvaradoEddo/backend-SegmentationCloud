import warnings

import torch
import time
import os
import numpy as np
from pathlib import Path
from PIL import Image
import helper
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from torchvision import datasets
from torchvision import datasets, transforms, models
from torch import nn, optim, Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
from tqdm import tqdm
import cv2
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cpu')
print('Using device:', device)
print()

seed=42
torch.manual_seed(14)

class RGBCloudDataset (Dataset):
    def __init__(self, red_dir, blue_dir, green_dir, gt_dir, n_samples=10000,transform= None):
        self.transform   = transform

        self.files = [self.combine_files(f, green_dir, blue_dir,gt_dir) 
                      for f in red_dir.iterdir() if not f.is_dir()]
        
        random.seed (seed)
        self.files = random.sample (self.files, k= n_samples)   
        
    def combine_files(self, red_file: Path, green_dir, blue_dir,  gt_dir):
        
        files = {'red': red_file, 
                 'green':green_dir/red_file.name.replace('red', 'green'),
                 'blue': blue_dir/red_file.name.replace('red', 'blue'), 
                 'gt': gt_dir/red_file.name.replace('red', 'gt')}

        return files
    
    
    
    def OpenAsArray(self, idx, invert=False):
        
        raw_rgb=np.stack([np.array(Image.open(self.files[idx]['red'])),
                          np.array(Image.open(self.files[idx]['green'])),
                          np.array(Image.open(self.files[idx]['blue']))], axis = 2)
     

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
    
    
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    
    
    
    def OpenMask(self, idx, add_dims=False):
        # print(self.files[idx]['gt'])
        raw_mask=np.array(Image.open(self.files[idx]['gt']))
        # raw_mask = np.where(raw_mask==255, 1, 0)
        
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask



        
    def __len__(self):
        
        return len(self.files)
    
    
    
    def __getitem__(self, idx):
        
        x=self.OpenAsArray(idx, invert=True, )
        y=self.OpenMask(idx, add_dims=False)

        if self.transform is not None:
            x , y= self.transform((x,y))
            

        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.float32)
        
        return x, y
    
    
    
    def open_as_pil(self, idx):
        
        arr = 256 * self.OpenAsArray(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')  
    
    
    
    def __repr__(self):
        
        s = 'Dataset class with {} files'.format(self.__len__())

        return s

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )

class UpsamplerBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, in_conv_channels=None, out_conv_channels=None):
        super().__init__()
        if (not in_conv_channels):
          in_conv_channels = in_channels
        if (not out_conv_channels):
          out_conv_channels = out_channels
        self.double_conv = double_conv(in_conv_channels + out_conv_channels, in_conv_channels)
        self.upsample =  up_conv(in_channels, out_channels)
    
    def forward(self, up_x, down_x=None):
        x = self.double_conv(up_x)
        x = self.upsample(x)
        if(down_x != None):
          x = torch.cat([x, down_x],1)
        return x

class ResNet34FCN_384(nn.Module):

    def __init__(self, *, out_channels=2,pretrained = True):
        super().__init__()
        self.encoder = models.resnet34(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(512, 512)

        self.up_block7 = UpsamplerBlockWithSkip(512,256)
        self.up_block8 = UpsamplerBlockWithSkip(256,128)
        self.up_block9 = UpsamplerBlockWithSkip(128,64)
        self.up_block10 = UpsamplerBlockWithSkip(64,32,64,64)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        if not pretrained:
            self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        x = self.up_conv6(block5)
        x = torch.cat([x, block4], dim=1)

        x = self.up_block7(x,block3)

        x = self.up_block8(x,block2)
        
        x = self.up_block9(x,block1)
        
        x = self.up_block10(x)
        
        x = self.out(x)

        return x

def get_model():
    ResNet_model2 = ResNet34FCN_384(pretrained=False).to(device)
    ResNet_model2.load_state_dict(torch.load('./trainet_5k.pt'))
    ResNet_model2.eval()
    return ResNet_model2

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import base64
import numpy as np
import base64

from io import BytesIO
def get_img_proccessed(image,model):
    output= np.zeros((1,2,256,256))
    with torch.no_grad():
        img=Image.open(image)
        x = TF.to_tensor(img)
        x.unsqueeze_(0)
        output=model(x)
        output=torch.round(output.softmax(dim=1))
        output = output.cpu().numpy()
    return output

def get_base64_img_from_mask(mask):
    mask_rgb = mask * 255
    mask_rgb = Image.fromarray(mask_rgb)
    if mask_rgb.mode != 'RGB':
        mask_rgb = mask_rgb.convert('RGB')

    buffered = BytesIO()
    mask_rgb.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str