import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy
import pandas

#parameters 
weight = 0.7 #b
bias = 0.3   #a

start = 0
end = 1
step = 0.01
x = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * x + bias

print(x[:10], y[:10])



