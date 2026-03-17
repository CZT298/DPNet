 # -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:03:14 2024

@author: S4300F
"""
# from torchvision.transforms import transforms as T
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


from model.Unet3d import UNet
from model.I2I3D import i2i
from model.vnet import Vnet
from model.unetr import UNETR
from model.Unet_plus_plus_3d import UNet as UNetPlus
from model.nnUnet import initialize_network as nnUNet
from model.transunet_3d import TransUNet
from model.swinUnetr import SwinUNETR
from model.resnet3d import resnet50
from model.AGUnet import AGUNet
from model.UXNet.network_backbone import UXNET
from model.ParaTransCNN3d import ParaVnetr
# from model.ParaSwinUNETR import ParaSwinUNETR
from model.SwinUnet3D import swinUnet_t_3D as swinunet
from model.WNet import WNet
from model.UNet_3layer import UNet as UNet3
from model.swin_unetr import SwinUNETR as SwinUNETR3
from model.csnet_3d import CSNet3D
from model.densevoxnet_torch import DenseVoxNet
from model.mednextv1.MedNextV1 import  MedNeXt


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#from dataload import train_dataload
import torch.optim as optim 
from dataload_3d import train_dataload , val_dataload
from torch.utils.data import DataLoader
from utils.dice import dice_score
# from utils.loss import BCELoss,DiceCELoss
import gc
from datetime import datetime
from monai.losses import DiceCELoss, DiceLoss
from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import swin_unetr
torch.autograd.set_detect_anomaly(True)
import os
import random
# random.seed(123)
torch.autograd.set_detect_anomaly(True)
def train_model(model,  optimizer, train_loader, num_epochs=100):
    # 将历史最小的loss（取值范围是[0,1]）初始化为最大值1
    # min_loss = 1
    max_test_dice = 0
    #critrion = torch.nn.BCELoss()#.to(device)
    for epoch in range(num_epochs):
        # 3个epoch不优化则降低学习率
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                                   threshold=0.0003, threshold_mode='rel', cooldown=50, min_lr=0,
                                                   eps=1e-04)
        epoch_iterator = tqdm( #进度条
           train_loader, desc="Training (X / X Steps) (loss=X.X,dice=X.X)", dynamic_ncols=True
       )
        epoch_loss = 0
        epoch_dice = 0
        for step,batch in enumerate(epoch_iterator):
            # x, y = batch['image'].float(), batch['label'].float().to('cuda:1')#(1,1,512,512,?)
            x, y = batch['image'].float().to(device), batch['label'].float().to(device)
            # step_loss = 0
            # step_dice = 0
            # print(1)
            optimizer.zero_grad() 
            out = model(x)       
            # print(out.shape,1)  
            # critrion = DiceLoss()
            critrion = DiceCELoss()
            loss =  critrion(y,out) 
            # print(loss.item())
            # print("11111111111111111111111111")
            loss.backward()
            # print("11111111111111111111111111")
            optimizer.step()
            dice = dice_score(y, out)
            # print(dice)
            epoch_dice += dice
            epoch_loss += loss.item()
                    
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (loss=%5f dice=%2.5f)" % (
                    epoch+1, step+1, len(train_loader), loss.item(),dice)
            )
            #gc.collect()
            #torch.cuda.empty_cache()
        ave_loss = epoch_loss / len(train_loader) 
        ave_dice = epoch_dice / len(train_loader)
        log(f'Epoch {epoch+1}/{num_epochs}, loss:{ave_loss:.5f},dice:{ave_dice:.5f}')
        print(f'Epoch {epoch+1}/{num_epochs}, loss:{ave_loss:.5f},dice:{ave_dice:.5f}') 
        # if min_loss > ave_loss: 
        torch.save(model.state_dict(), pth)            
            
        
        if (epoch+1) % 5 == 0:
            test_dice = test(img_size)
            if test_dice > max_test_dice:
                torch.save(model.state_dict(), test_pth )
                max_test_dice = test_dice
    return model     
    
# 训练模型
def train(batch_size, img_size):
    #log时间
    log(f'\nNOW TIME:{datetime.now()}')
    print(f'NOW TIME:{datetime.now()}')

    # 导入历史保存的权重作为训练初始权重
    # model.load_state_dict(torch.load(load_pth, map_location='cpu'),strict=False)  # JY11.21,加载之前的训练结果，到model中
    # batch_size设为1
    
    
    # 梯度下降的优化器，使用默认学习率
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    train_loader = train_dataload(batch_size,img_size)
    # 开始训练
    # print(1)
    train_model(model,  optimizer, train_loader)
    
def test(img_size): 
    batch_size = 1
    val_loader = val_dataload(batch_size)
    epoch_iterator = tqdm( #进度条
                          val_loader, desc="Testing (X / X Steps) (val_dice=X.X)", dynamic_ncols=True
                          )
    epoch_dice = 0
    with torch.no_grad():
        for step,batch in enumerate(epoch_iterator):
            x, y = batch['image'].float().to(device), batch['label'].float().to(device)#(1,1,512,512,?)
            # x, y = batch['image'].float(), batch['label'].float().to('cuda:1')
            # step_dice = 0
        
            out = sliding_window_inference(x, img_size, sw_batch_size=1, predictor= model, overlap=0.2)
            dice = dice_score(y,out)
            epoch_iterator.set_description(
            "Testing (%d / %d Steps) (val_dice=%2.5f)" % (
            step, len(val_loader), dice)
            )       
            epoch_dice += dice
   
    ave_dice = epoch_dice / len(val_loader)
    log(f'valdiation ave_val_dice:{ave_dice:.5f}')
    print(f'valdiation ave_val_dice:{ave_dice:.5f}') 
    
    return  ave_dice    

def log(str):
    f = open(log_path, 'a')
    f.write(str + '\n')
    f.close()
    
if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    rank = dist.get_rank() 
    
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    load_pth= './pth_128/val_best/ours_vessel_val.pth'
    pth = './pth_128/train/ours_vessel.pth'
    test_pth = './pth_128/val_best/ours_vessel_val.pth'
    log_path = './trainning_log/ours.txt'
    
    img_size = (128,128,128)
    batch_size = 1
    # model = resnet50(
    #         sample_input_W=128,
    #         sample_input_H=128,
    #         sample_input_D=128,
    #         shortcut_type='B',
    #         no_cuda=False,
    #         num_seg_classes=1)
    
    
    # model = UNETR(in_channels=1,
    #                 out_channels=1,
    #                 img_size=img_size,
    #                 feature_size=16,
    #                 hidden_size=768,
    #                 mlp_dim=3072,
    #                 num_heads=12,
    #                 pos_embed='perceptron',
    #                 norm_name='instance',
    #                 conv_block=True,
    #                 res_block=True,
    #                 dropout_rate=0.0)
    # model = UNet(1,1)
    # model = TransUNet(img_dim=(128, 128, 128),
    #                       in_channels=1,
    #                       out_channels=128,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=1)
    
    # model = SwinUNETR3(img_size = img_size,
    #                       in_channels=1,
    #                       out_channels=1)
    # model = AGUNet(1,1)
    # model = ParaVnetr(in_channels=1,
    #                 out_channels=1,
    #                 img_size=(128, 128, 128),
    #                 feature_size=16,
    #                 hidden_size=256,
    #                 mlp_dim=3072,
    #                 num_heads=16,
    #                 pos_embed='perceptron',
    #                 norm_name='instance',
    #                 conv_block=True,
    #                 res_block=True,
    #                 dropout_rate=0.0)
    model = WNet(img_size =(128,128,128) ,
                          in_channels=1,
                          out_channels=1,
                          lgff = 1,
                          merge = 1)
    
    # model = CSNet3D(1, 1)
    # model = DenseVoxNet(n_classes=1)
    # model = MedNeXt(
    #         in_channels = 1, 
    #         n_channels = 32,
    #         n_classes = 1,
    #         # exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         exp_r = 2,
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         do_res_up_down = True,
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #         # block_counts = [3,4,8,8,8,8,8,4,3],
    #         checkpoint_style = None,
    #         dim = '3d',
    #         grn=True
            
    #     )
    
    
    
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     for k in ['decoder5', 'decoder4', 'decoder3', 'decoder2', 'decoder1', 'out']:
    #         if str(k) in name:
    #             param.requires_grad = True
                
    # model = UXNET(in_chans=1,out_chans=1)
    
    # window_size = [i // 32 for i in [128,128,128]]
    # print(window_size)
    # model = swinunet(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
    #                     window_size=window_size, in_channel=1, num_classes=1
    #                     )
    # model = nn.DataParallel(model)
    model = model.to(device)
    model = DDP(model, device_ids=[rank],output_device=rank,find_unused_parameters=True)
    # 
    
    # 参数解析
    with torch.cuda.device(0):
        # 训练
        
        train(batch_size,img_size)
        # test(img_size)
        

        
        
        
        