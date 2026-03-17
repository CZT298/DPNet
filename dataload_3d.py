# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:34:14 2024

@author: S4300F
"""
# from nibabel.nifti1 import Nifti1Image
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
# import numpy as np
# import torch
from monai.transforms import Compose,LoadImaged, AddChanneld, ToTensord,CropForegroundd,RandSpatialCropd
    # AsDiscrete,
   
from monai.transforms import  Orientationd,RandGaussianNoised,RandScaleIntensityd,RandShiftIntensityd,ResizeWithPadOrCropd
    
    
    # Orientationd,
    # RandFlipd,
    # RandCropByPosNegLabeld,
    # RandShiftIntensityd,
    # ScaleIntensityRanged,
    # Spacingd,
    # RandRotate90d,
   
    # CenterSpatialCropd,
    # Resized,
    # SpatialPadd,
    # apply_transform,
    
    
    
    # ResizeWithPadOrCropd
    



# from monai.data import CacheDataset, DataLoader
# import collections.abc
# import math
# import pickle
# import shutil
import sys
# import tempfile
# import threading
# import time
# import warnings
# from copy import copy, deepcopy

import numpy as np

# import numpy as np
# import torch
# from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
# import random
# random.seed(123)
from tqdm import tqdm
# from torch.utils.data import Subset
import nibabel as nib
import random
# random.seed(123)
# from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
# from monai.config import DtypeLike, KeysCollection
# from monai.transforms.transform import Transform, MapTransform
# from monai.utils.enums import TransformBackends
# from monai.config.type_definitions import NdarrayOrTensor
# from monai.transforms.io.array import LoadImage, SaveImage
# from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
# from monai.data.image_reader import ImageReader
# from monai.utils.enums import PostFix
# from torch.utils.data import ConcatDataset
# from torch.utils.data import DataLoader as DataLoaders
# DEFAULT_POST_FIX = PostFix.meta()
# import os
# import h5py
# import torch.utils.data as data
# NUM_CLASS = 3
dataset_dir = '../data/'
data_txt_path = 'dataset_list'

def train_dataload(batch_size,img_size):
    train_img = []
    train_lbl = []
    train_name = []
    
    for line in open(data_txt_path + '/train_ASOCA.txt'):
        train_img.append(dataset_dir + line.strip().split()[0])#.replace('img_cut', 'img_cut_fat'))#img_cut_fat
        #train_lbl.append(dataset_dir + line.strip().split()[1])#脂肪 
        train_lbl.append(dataset_dir + line.strip().split()[1]) #血管
    data_dicts_train = [{'image': image, 'label': label}
                for image, label in zip(train_img, train_lbl)]
    
    train_transforms = Compose([
            LoadImaged(keys=["image", "label"]), #
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            
            CropForegroundd(keys=["image", "label"],source_key='label'),#去掉label背景
            # # RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=2, spatial_axes=(2,1)),
            # # Spacingd(keys=['image','label'], pixdim=(1.5,1.5,2)),
            # # ScaleIntensityRanged(keys='image',a_min=-200,a_max=200,b_min=0.0,b_max=1.0,clip=True),
            # # RandCropByPosNegLabeld(keys=["image", "label"],label_key='label',num_samples=4,spatial_size=(img_size)),
            
            RandSpatialCropd(
                            keys=["image", "label"],
                            roi_size=img_size,
                            random_size=False
                                            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=img_size),
            # # Resized(keys = ["image", "label"], spatial_size=(256,256,128)),
            RandGaussianNoised(keys="image", prob=0.5, std=0.05),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )
    # print(1)
    train_dataset = Dataset(data=data_dicts_train,transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,sampler=DistributedSampler(train_dataset))

    return train_loader
def val_dataload(batch_size):
    val_img = []
    val_lbl = []
    val_name = []
    for line in open(data_txt_path + '/val_ASOCA.txt'):
        val_img.append(dataset_dir + line.strip().split()[0])#.replace('img_cut', 'img_cut_fat'))
        #val_lbl.append(dataset_dir + line.strip().split()[1]) #脂肪
        val_lbl.append(dataset_dir + line.strip().split()[1])               
        #val_name.append(line.strip().split()[0].split('.')[0])
        
        #test_lbl.append(dataset_dir + line.strip().split()[2]) #血管
    data_dicts_val = [{'image': image, 'label': label}
                for image, label in zip(val_img, val_lbl)]
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            CropForegroundd(keys=["image", "label"],source_key='label'),#去掉image背景
            # RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=2, spatial_axes=(2,1)),
            # Spacingd(keys=['image','label'], pixdim=(1.5,1.5,2)),
            # ScaleIntensityRanged(keys='image',a_min=-200,a_max=200,b_min=0.0,b_max=1.0,clip=True),
            # Resized(keys = ["image", "label"], spatial_size=(256,256,128)),
            RandGaussianNoised(keys="image", prob=0.5, std=0.05),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_dataset = Dataset(data=data_dicts_val,transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,sampler=DistributedSampler(val_dataset))
    return val_loader

def test_dataload(batch_size):
    test_img = []
    test_lbl = []
    test_name= []
  
    affine = []
    for line in open(data_txt_path + '/test_ASOCA.txt'):
        test_img.append(dataset_dir + line.strip().split()[0])#.replace('img_cut', 'img_vnet_cut'))
        #test_lbl.append(dataset_dir + line.strip().split()[1]) 
        test_lbl.append(dataset_dir + line.strip().split()[1])
        #test_name.append(line.strip().split()[0])
        test_name.append(line.strip().split()[0])

        
        
    for i in test_img:
        img_y = nib.load(i)
        affine.append(img_y.affine)
        
    data_dicts_test = [{'image': image, 'label': label, 'name':name, 'affine':affine}
                for image, label, name, affine in zip(test_img, test_lbl, test_name,affine)]
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            CropForegroundd(keys=["image", "label"],source_key='label'),#去掉image背景
            # Spacingd(keys=['image','label'], pixdim=(1.5,1.5,2)),
            # ScaleIntensityRanged(keys='image',a_min=-200,a_max=200,b_min=0.0,b_max=1.0,clip=True),
            RandGaussianNoised(keys="image", prob=0.5, std=0.05),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )   
    test_dataset = Dataset(data=data_dicts_test,transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader 
 
def test_dataload_nn(batch_size):
    test_img = []
    test_lbl = []
    test_name= []
  
    affine = []
    for line in open(data_txt_path + '/nnunettest.txt'):
        test_img.append(dataset_dir + line.strip().split()[0])#.replace('img_cut', 'img_vnet_cut'))
        #test_lbl.append(dataset_dir + line.strip().split()[1]) 
        test_lbl.append(dataset_dir + line.strip().split()[1])
        #test_name.append(line.strip().split()[0])
        test_name.append(line.strip().split()[1])

        
        
    for i in test_img:
        img_y = nib.load(i)
        affine.append(img_y.affine)
        
    data_dicts_test = [{'image': image, 'label': label, 'name':name, 'affine':affine}
                for image, label, name, affine in zip(test_img, test_lbl, test_name,affine)]
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            # CropForegroundd(keys=["image", "label"],source_key='label'),#去掉image背景
            # Spacingd(keys=['image','label'], pixdim=(1.5,1.5,2)),
            # ScaleIntensityRanged(keys='image',a_min=-200,a_max=200,b_min=0.0,b_max=1.0,clip=True),
            # RandGaussianNoised(keys="image", prob=0.5, std=0.05),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )   
    test_dataset = Dataset(data=data_dicts_test,transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader         
if __name__ == '__main__':
    # dist.init_process_group(backend="gloo", rank=0, world_size=1)
    train_loader = train_dataload(1,(1,1,128,128,128))
    iterator = tqdm( #进度条
                          train_loader, desc="", dynamic_ncols=True
                          )
    for index, batch in enumerate(iterator):
        # x, y,name,affine= batch["image"].float(), batch["label"].float(),batch["name"][0],batch["affine"][0]
        x, y = batch["image"].float(), batch["label"].float()
        # save_path = 'out/ASOCA/image/' +  name.strip().split('/')[-1]
        # out = x.numpy().squeeze(0).squeeze(0)
        # out = Nifti1Image(out,affine)
        # nib.save(out, save_path)
        print(x.shape,y.shape)

