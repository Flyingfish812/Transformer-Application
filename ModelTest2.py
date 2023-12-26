import numpy as np
import torch
from Sample import *
from torch.nn import Transformer
from torchvision.models import vision_transformer as vit

image_size = 180*360
patch_size = 18
num_layers = 12
num_heads = 12
hidden_dim = 768
mlp_dim = 3072

model = vit.VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    mlp_dim=mlp_dim,
)

file_name = './sst_weekly.mat'
lat,lon,sst_all,time = read_hdf5(file_name)
sst = sst_all[0]
print(sst.shape)