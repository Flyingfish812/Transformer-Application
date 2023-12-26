import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from Sample import *
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vision_transformer as vit

# Read the file
f = h5py.File('./sst_weekly.mat','r')
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst_all = np.array(f['sst'])
time = np.array(f['time'])

