# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:16:56 2024

@author: S4300F
"""

import torch
from torch import nn
# import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from model.Unet3d import UNet
from model.I2I3D import i2i
from model.vnet import Vnet
from model.unetr import UNETR
from model.Unet_plus_plus_3d import UNet as UNetPlus
from model.nnUnet import initialize_network as nnUNet
from model.transunet_3d import TransUNet
from model.swinUnetr import SwinUNETR
from model.resnet3d import resnet50
from model.UXNet.network_backbone import UXNET
from model.AGUnet import AGUNet
from model.ParaTransCNN3d import ParaVnetr
# from model.ParaSwinUNETR import ParaSwinUNETR
from model.SwinUnet3D import swinUnet_t_3D as swinunet
from model.WNet import WNet
from model.UNet_3layer import UNet as UNet3
from model.swin_unetr import SwinUNETR as SwinUNETR3
from model.csnet_3d import CSNet3D
from model.densevoxnet_torch import DenseVoxNet
from model.mednextv1.MedNextV1 import  MedNeXt



import warnings
warnings.filterwarnings("ignore")
# from tensorboardX import SummaryWriter
# import random
# random.seed(123)
# from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference
# from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
# from monai.transforms import AsDiscrete
# from monai.metrics import DiceMetric
# from dataset.dataloader import get_loader

# from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS, ORGAN_NAME
# from torchvision.transforms import transforms as T
from dataload_3d import test_dataload,test_dataload_nn
#from dataload_test import test_dataload
# import torch.optim as optim 
# from utils.dice import dice_score
# import os
# import SimpleITK as sitk
# from torch.utils.data import DataLoader
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from monai.inferers import sliding_window_inference
# import torchvision.ops as ops
# import cv2
from scipy.ndimage import morphology
from model.unetr import UNETR
from medpy import metric
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 



def saveOut(out,y,name,affine):
    
    # save_path = 'out/CAS/' + name.strip().split('/')[-1].split('_')[0] + '/mednext_'+ name.strip().split('/')[-1].split('_')[0] + '_vessel_out.nii.gz' 
    # # save_path = 'out/CAS/nnunet_' + name.strip().split('/')[-1].split('_')[0] + '_vessel_out.nii.gz' 
    save_path = 'out/ASOCA/model/csnet/csnet_' + name.strip().split('/')[-1].split('_')[0] + '_vessel_'+ name.strip().split('/')[-1].split('_')[-1]
    out = out.cpu().detach().numpy()#(1,1,512,512,?)
    ##out = out.transpose(1,2,3,4,0)
    out = out.squeeze(0).squeeze(0)
  
    out = Nifti1Image(out, affine)
    nib.save(out, save_path)
    
    # save_path_y = 'out/y/' + name.strip().split('/')[-1].split('_')[0] + '_out_y.nii.gz'
    # save_path_y = 'out/y/' + name.strip().split('/')[-1].split('_')[0] + '_y_' + name.strip().split('/')[-1].split('_')[-1]
    # y = y.cpu().detach().numpy()#(1,1,512,512,?)
    # y = y.squeeze(0).squeeze(0)
    # y = Nifti1Image(y, affine)
    # nib.save(y, save_path_y)  
    
    # save_path_x =  'out/img/' + name.strip().split('/')[-1].split('_')[0] + '_out.nii.gz'
    # x = y.cpu().detach().numpy()#(1,1,512,512,?)
    # x = x.squeeze(0).squeeze(0)
    # x = Nifti1Image(x, affine)
    # nib.save(x, save_path_x)
    
def calculate_metric_percase(output, label):
    pred = output.cpu().numpy()
    gt = label.cpu().numpy()

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # acc = metric.binary.precision(pred, gt)
    # recall = metric.binary.recall(pred, gt)\\\\\
    # # spe = metric.binary.specificity(pred, gt)
    # intersection = np.logical_and(pred,gt)\\\\
    # union = np.logical_or(pred,gt)
    # iou = np.mean(np.sum(intersection)/np.sum(union))
    return dice, hd, jc

def detect_outliers(data, m=1.5):
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (m * IQR)
    upper_bound = Q3 + (m * IQR)
    over = data[(data < lower_bound) | (data > upper_bound)]
    right = data[data > lower_bound]
    return over, right

def result_count(dice_list, hd_list, jc_list):
    
    anwer_list=[]
    
    dice_ave = statistics.mean(dice_list)
    dice_r = statistics.stdev(dice_list)
    
    jc_ave = statistics.mean(jc_list)
    jc_r = statistics.stdev(jc_list)
    
    hd_ave = statistics.mean(hd_list)
    hd_r = statistics.stdev(hd_list)
    

    
    print("\nAVERAGE DICE SCORE:%2.5f r:%2.5f\n" % (dice_ave,dice_r))
    print("AVERAGE JC SCORE:%2.5f r:%2.5f\n" % (jc_ave,jc_r))
    print("AVERAGE HD95 SCORE:%2.5f r:%2.5f\n" % (hd_ave,hd_r))
      
    dice_over, dice_right = detect_outliers(dice_list)
    jc_over, jc_right = detect_outliers(jc_list)
    hd_over, hd_right = detect_outliers(hd_list)

    
    dice_right_ave = statistics.mean(dice_right)
    dice_right_r = statistics.stdev(dice_right)
    anwer_list.append("%2.5f±%2.5f" % (dice_right_ave,dice_right_r))
    
    hd_right_ave = statistics.mean(hd_right)
    hd_right_r = statistics.stdev(hd_right)
    anwer_list.append("%2.5f±%2.5f" % (hd_right_ave,hd_right_r))
    
    jc_right_ave = statistics.mean(jc_right)
    jc_right_r = statistics.stdev(jc_right)
    anwer_list.append("%2.5f±%2.5f" % (jc_right_ave,jc_right_r))
    print('*******************************************************************')
    print("AVERAGE DICE SCORE AFTER BOXING:%2.5f r:%2.5f\n" % (dice_right_ave,dice_right_r))
    print("AVERAGE JC SCORE AFTER BOXING:%2.5f r:%2.5f\n" % (jc_right_ave,jc_right_r))
    print("AVERAGE HD95 SCORE AFTER BOXING:%2.5f r:%2.5f\n" % (hd_right_ave,hd_right_r))
   
    
    return anwer_list
    

def test():   
    # model.eval()
    # model = resnet50(
    #         sample_input_W=128,
    #         sample_input_H=128,
    #         sample_input_D=128,
    #         shortcut_type='B',
    #         no_cuda=False,
    #         num_seg_classes=1)
    # model = Vnet(1,1)
    # model = UNet3(1,1)
    # model = UNetPlus(1,1)
    # model = UXNET(in_chans=1,out_chans=1)
    # model = UNETR(in_channels=1,
    #                 out_channels=1,
    #                 img_size=(128,128,128),
    #                 feature_size=16,
    #                 hidden_size=768,
    #                 mlp_dim=3072,
    #                 num_heads=12,
    #                 pos_embed='perceptron',
    #                 norm_name='instance',
    #                 conv_block=True,
    #                 res_block=True,
    #                 dropout_rate=0.0)

    # model = SwinUNETR3(img_size = (128,128,128),
    #                       in_channels=1,
    #                       out_channels=1)
    
    # model = TransUNet(img_dim=(128, 128, 128),
    #                       in_channels=1,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=1)
   
    
    # model = AGUNet(1,1)

    # model = WNet(img_size =(128,128,128),
    #                       in_channels=1,
    #                       out_channels=1,
    #                       lgff = 1,
    #                       merge = 1)
    # model = ParaSwinUNETR(img_size =(128,128,128) ,
    #                       in_channels=1,
    #                       out_channels=1,
    #                       )
    # window_size = [i // 32 for i in [128,128,128]]
    # # print(window_size)
    # model = swinunet(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
    #                     window_size=window_size, in_channel=1, num_classes=1
    #                     )
    
    # 
    model = CSNet3D(1, 1)
    # model = DenseVoxNet(n_classes=1)
    # model = MedNeXt(
    #         in_channels = 1, 
    #         n_channels = 32,
    #         n_classes = 1,
    #         exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         #exp_r = 2,
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         do_res_up_down = True,
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #         # block_counts = [3,4,8,8,8,8,8,4,3],
    #         checkpoint_style = None,
    #         dim = '3d',
    #         grn=True
    #         )
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load('./pth_ASOCA/val_best/csnet_vessel_val.pth'),strict=False) #train best
    # model.load_state_dict(torch.load('./pth_128/val_best/uxnet_vessel_val.pth'))  # test best
    # model.load_state_dict(torch.load('./pth_128/val_best/paraswinunetr_vessel_val.pth')) 
    #model.eval()  # 设置模型为评估模式  
    
    # 如果需要，准备测试集  
    test_loader = test_dataload(batch_size=1)
    #liver_dataset = test_dataload(batch_size=1,transform = x_transform, target_transform = y_transform)
    #test_loader = DataLoader(liver_dataset, batch_size=1)
    iterator = tqdm( test_loader, desc="(dice=X.X)", dynamic_ncols=True)
    # dice_pic = 0.0

    dice_list,  hd_list,jc_list = [],[],[]
    # model.eval()
    with torch.no_grad(): 
        
        for step, batch in enumerate(iterator):        
            x, y,name,affine= batch["image"].float().to(device),batch["label"].float().to(device),batch["name"][0],batch["affine"][0]
            
            out = sliding_window_inference(x, (128,128,128), sw_batch_size=2, predictor= model, overlap=0.5)
            out = torch.where(out <= 0.5, torch.tensor(0, dtype=torch.float).to(device), out)
            out = torch.where(out > 0.5, torch.tensor(1, dtype=torch.float).to(device), out)
            
            # kernel1 = np.ones((1,1,1), np.uint8) 
            # kernel2 = np.ones((1,1,1), np.uint8) 
            
            
            # out = morphology.binary_opening(out.cpu().detach().numpy().squeeze(0).squeeze(0), structure=kernel1)
            # out = morphology.binary_opening(out, structure=kernel2)
            # out = torch.tensor(out)
            # out = out.unsqueeze(0).unsqueeze(0).to(device).float()
            
            dice, hd, jc = calculate_metric_percase(out, y)
               
            iterator.set_description("(dice=%2.5f)" % dice)
            
            dice_list.append(dice)
            jc_list.append(jc)
            hd_list.append(hd)
           
            #print(out.shape,y.shape)
            #print(name[0])
            saveOut(out ,y,name,affine)
            print(name)
        anwer_list = result_count(dice_list, hd_list, jc_list)
        
    return anwer_list,dice_list,  hd_list, jc_list

def nnunet_test():
    dice_list,  hd_list, jc_list = [],[],[]
    test_loader = test_dataload_nn(batch_size=1)
    iterator = tqdm( test_loader, desc="(dice=X.X)", dynamic_ncols=True)
    with torch.no_grad(): 
        for step, batch in enumerate(iterator):
            x, y,name,affine= batch["image"].float().to(device),batch["label"].float().to(device),batch["name"][0],batch["affine"][0]
            dice,  hd, jc= calculate_metric_percase(x, y)
            iterator.set_description("(dice=%2.5f)" % dice)
            dice_list.append(dice)
            jc_list.append(jc)
            hd_list.append(hd)
            
            #print(out.shape,y.shape)
            #print(name[0])
            #saveOut(x ,y,name,affine)
            print(name)
        anwer_list = result_count(dice_list, hd_list, jc_list)
    
        return anwer_list,dice_list,  hd_list, jc_list
            
            
            
            
            
            
            
            
            
            
if __name__ == '__main__':
    with torch.cuda.device(0):
        anwer_list,dice_list,  hd_list,jc_list = test()
        # anwer_list,dice_list,  hd_list, jc_list = nnunet_test()
        # test()
    