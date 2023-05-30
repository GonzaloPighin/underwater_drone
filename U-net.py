#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:57:58 2023

@author: gonzalo
"""
import os
#from dataset import CarvanaDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
#from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

# Hyperparameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/gonzalo/Escritorio/dist&surf/UNET/foto_entrenamiento"
TRAIN_MASK_DIR = "/home/gonzalo/Escritorio/dist&surf/UNET/máscara_entrenamiento"
VAL_IMG_DIR = "/home/gonzalo/Escritorio/dist&surf/UNET/foto_validación"
VAL_MASK_DIR = "/home/gonzalo/Escritorio/dist&surf/UNET/máscara_validación"

###

# Class definition

class LeakDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(self.image_dir)
        self.mask_list = os.listdir(self.mask_dir)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir,self.image_list[index])
        mask_path = os.path.join(self.mask_dir,self.mask_list[index])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = np.array(img)
        mask = np.array(mask)
        mask[mask==255.0] = 1.0
        #img_mask_dict = {"image": img, "mask": mask}
        
        if self.transform:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]
            mask = torch.unsqueeze(mask,0)
            #transformations = self.transform(image=img, mask=mask)
            #img = transformations["image"]
            #mask = transformations["mask"]
            
        return img,mask

###

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True):
    
    train_ds = LeakDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)
    
    val_ds = LeakDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False)
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)
            
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}") 
    print(f"Dice score: {dice_score/len(loader)}") 
    model.train()
    
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"")

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.conv(x)
    
class UNET(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Up part of UNET
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2,feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self,x):
        
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
            
        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.final_conv(x)

"""             
def test():
    x = torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    """
    
class UWDDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        
def main():
    
    train_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                 A.Rotate(limit=35, p=1.0),
                                 A.HorizontalFlip(p=0.5),
                                 A.VerticalFlip(p=0.1),
                                 A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std= [1.0, 1.0, 1.0],
                                             max_pixel_value=255.0),
                                             ToTensorV2()])
    
    val_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                               A.Normalize(mean = [0.0, 0.0, 0.0], 
                                           std = [1.0, 1.0, 1.0], 
                                           max_pixel_value = 255.0),
                                           ToTensorV2()])
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # crossentropyloss sino
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {"state_dict": model.state_dict(), 
                      "optimizer": optimizer.state_dict()}
        
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(val_loader, model, folder="/home/gonzalo/Escritorio/dist&surf/UNET/resultados", device=DEVICE)
        
    
if __name__=='__main__':
    main()