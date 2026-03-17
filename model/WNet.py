# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:04:04 2024

@author: zyserver
"""

import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from typing import Tuple, Union
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from .swin_unetr import SwinTransformer,PatchMerging,PatchMergingV2
from typing import Optional, Sequence, Tuple, Type, Union
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
# from pytorch_wavelets import DWTForward
import numpy as np



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UnetEncoder(nn.Module):
    def __init__(self,in_channels,feature_size):
        super(UnetEncoder,self).__init__()
        size = feature_size
        self.conv0 = DoubleConv(in_channels, 1 * size)
        self.conv1 = DoubleConv(1 * size, 1 * size)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv(1 * size, 2 * size)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv(2 * size, 4 * size)
        self.pool3 = nn.MaxPool3d(2)
        # self.conv4 = DoubleConv(4 * size, 8 * size)
        # self.pool4 = nn.MaxPool3d(2)


    def forward(self,x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        # x4 = self.conv4(p3)
        # p4 = self.pool4(x4)
        return x0,p1,p2,p3#,p4
       
class LGFF(nn.Module):
    def __init__(self,channel, r=16):
        super(LGFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel//2, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(channel//2),
            nn.ReLU(inplace=True),
        )


    def forward(self, c_in,st_in):
        x = torch.cat([c_in, st_in], dim=1)
        b, c, _, _,_ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        y = self.conv(y)
        c_out = c_in + y
        st_out = st_in + y
        return c_out, st_out   
    
class LGFF2(nn.Module):
    def __init__(self,channel, r=16):
        super(LGFF2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, c_in,st_in):
      
        b, c, _, _,_ = c_in.size()
        # Squeeze
        c_y = self.avg_pool(c_in).view(b, c)
        # Excitation
        c_y = self.fc(c_y).view(b, c, 1, 1, 1)
        # Fusion
        st_ff = torch.mul(st_in, c_y)
        st_out = st_in + st_ff
        
        b, c, _, _,_ = st_in.size()
        # Squeeze
        st_y = self.avg_pool(st_in).view(b, c)
        # Excitation
        st_y = self.fc(st_y).view(b, c, 1, 1, 1)
        # Fusion
        c_ff = torch.mul(c_in, st_y)
        c_out = st_in + c_ff
        return c_out, st_out 
    
class LGFF0(nn.Module):
    def __init__(self):
        super(LGFF0, self).__init__()

    def forward(self, c_in,st_in):
        c_out = c_in
        st_out = st_in
        return c_out, st_out   
# class MERGE(nn.Module):
#     def __init__(self,channel):
#         super(MERGE, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         # self.linear = nn.Linear(channel*2, 1, bias=False)
#     def forward(self,c_in,st_in):
        
#         x = torch.cat([c_in, st_in], dim=1)
#         print(x.shape)
#         # x = self.linear(x)
#         z = self.sigmoid(x)
        
#         return z.view(z.size()[0], 1) * c_in + (1 - z).view(z.size()[0], 1) * st_in
class MERGE(nn.Module):
    def __init__(self,channel=48):
        super(MERGE, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, (3, 3, 3), padding=1), 
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, channel//2, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(channel//2),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, cin,stin):
        xin = torch.cat([cin,stin], dim=1)
        avgout = torch.mean(xin, dim=1, keepdim=True)
        maxout, _ = torch.max(xin, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        # print(x.shape)
        x = self.conv(x)
        x = self.sigmoid(x)#1,1,128,128,128
        # print(x.shape)
        out = xin * x 
        out = self.conv2(out)
        return out    

class MERGE2(nn.Module):
    def __init__(self):
        super(MERGE2, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, (3, 3, 3), padding=1), 
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, cin,stin):
        avgout1 = torch.mean(cin, dim=1, keepdim=True)
        maxout1, _ = torch.max(cin, dim=1, keepdim=True)
        c = torch.cat([avgout1, maxout1], dim=1)
        c = self.conv(c)
        c_sita = self.sigmoid(c)
        
        avgout2 = torch.mean(stin, dim=1, keepdim=True)
        maxout2, _ = torch.max(stin, dim=1, keepdim=True)
        st = torch.cat([avgout2, maxout2], dim=1)
        st = self.conv(st)
      
        cout = cin * c_sita
        stout = stin * (1-c_sita)
        
        return cout + stout 

class MERGE0(nn.Module):
    def __init__(self):
        super(MERGE0, self).__init__()
       
    def forward(self, cin,stin):
        out = cin + stin
        return out
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
class WNet(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        lgff = 1,
        merge = 1,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        """

        super(WNet,self).__init__()   
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)   
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.normalize = normalize
        self.UnetEncoder = UnetEncoder(in_channels=in_channels,feature_size=feature_size)
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )

        if lgff == 1:
            self.LGFF_1 = LGFF(feature_size*2)
            self.LGFF_2 = LGFF(feature_size*4)
        elif lgff == 2:
            self.LGFF_1 = LGFF2(feature_size)
            self.LGFF_2 = LGFF2(feature_size*2)
        elif lgff == 0:
            self.LGFF_1 = LGFF0()
            self.LGFF_2 = LGFF0()

        self.transenc1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.transenc2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.transenc3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.transenc4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.cnndec3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.cnndec2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.cnndec1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.transdec3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.transdec2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.transdec1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ) 
        
        if merge==1 :
            self.merge = MERGE(feature_size*2)
        if merge==2 :
            self.merge = MERGE2()
        if merge==0 :
            self.merge = MERGE0()
            
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in):
        c_enc0,u_out1,u_out2,u_out3 = self.UnetEncoder(x_in) 
        # print(u_out1.shape)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        # print(hidden_states_out[0].shape,hidden_states_out[1].shape,hidden_states_out[2].shape)
        st_enc0 = self.transenc1(x_in)
        # print(st_enc0.shape)
        # c_enc0, st_enc0 = self.LGFF_0(c_enc0, st_enc0)
        st1 = self.transenc2(hidden_states_out[0])

        c_enc1, st_enc1 = self.LGFF_1(u_out1,st1)
        
        st2= self.transenc3(hidden_states_out[1])
        c_enc2, st_enc2 = self.LGFF_2(u_out2,st2)
        
        st3 = self.transenc4(hidden_states_out[2])
        # c_enc3, st_enc3 = self.LGFF_3(u_out3,st3) 
        # print(c_enc3.shape,u_out2.shape)
        # print(u_out3.shape,c_enc2.shape)
        c_dec3 = self.cnndec3(u_out3,c_enc2)
        c_dec2 = self.cnndec2(c_dec3,c_enc1)
        c_dec1 = self.cnndec1(c_dec2,c_enc0)
        
        st_dec3 = self.transdec3(st3,st_enc2)
        st_dec2 = self.transdec2(st_dec3,st_enc1)
        st_dec1 = self.transdec1(st_dec2,st_enc0)

        # print(c_dec1.shape,st_dec1.shape)
        # out = self.merge(torch.cat([c_dec1,st_dec1], dim=1))
        out = self.merge(c_dec1,st_dec1)
        logits = self.out(out)
        
        return logits
    
if __name__ == '__main__':
    from torchsummary import summary
    
    model = WNet(img_size =(128,128,128) ,
                          in_channels=1,
                          out_channels=1,
                          lgff = 1,
                          merge = 1
                          )#.to('cuda')
    
    # mask = model(torch.rand(1, 1, 128,128,128))
    summary(model,(1,128,128,128),device='cpu')  
    
    from thop import profile, clever_format
    
    
    # params = 0
    # model = torch.load('../pth_128/val_best/wnet11_vessel_val.pth')
    # for k,v in model.items():#['network_weights']
    #     print(v.shape)
    #     tmp = 1
    #     print(len(v.shape))
    #     for i in range(len(v.shape)):
    #         tmp *= v.shape[i]
    #     params += tmp
    # print(params)
    
    # input = torch.rand(1, 1, 128,128,128)
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("FLOPs: %s" %(flops))
    # print("params: %s" %(params))
    
    # print('Number of layers in the base model: ', len(list(model.children())))
    # merge = MERGE2()  
    # out = merge(torch.rand(1, 24, 128,128,128),torch.rand(1, 24, 128,128,128))
       
    