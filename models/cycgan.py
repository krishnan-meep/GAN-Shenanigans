import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.building_blocks import ResBlock

class UNet_Generator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3):
        super(UNet_Generator, self).__init__()
        self.C1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)
        self.I1 = nn.InstanceNorm2d(32)
        
        self.C2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.I2 = nn.InstanceNorm2d(64)
        
        self.C3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.I3 = nn.InstanceNorm2d(128)
        
        self.C4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.I4 = nn.InstanceNorm2d(256)
        
        self.C5 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.I5 = nn.InstanceNorm2d(512)

        self.C6 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.I6 = nn.InstanceNorm2d(512)

        self.R1 = ResBlock(512, specnorm = False, batchnorm = True, use_instance_norm = True)
        self.R2 = ResBlock(512, specnorm = False, batchnorm = True, use_instance_norm = True)

        self.TC1 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.IT1 = nn.InstanceNorm2d(256)
        
        self.TC2 = nn.ConvTranspose2d(256 + 256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.IT2 = nn.InstanceNorm2d(128)
        
        self.TC3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.IT3 = nn.InstanceNorm2d(64)
        
        self.TC4 = nn.ConvTranspose2d(64 + 64, 32, kernel_size = 4, stride = 2, padding = 1)
        self.IT4 = nn.InstanceNorm2d(32)

        self.TC5 = nn.ConvTranspose2d(32 + 32, out_channels, kernel_size = 4, stride = 2, padding = 1)        
        
    def forward(self, x):
        a = F.leaky_relu(self.I1(self.C1(x)))   
        b = F.leaky_relu(self.I2(self.C2(a)))
        b = F.dropout(b, p = 0.35)
        c = F.leaky_relu(self.I3(self.C3(b)))   
        d = F.leaky_relu(self.I4(self.C4(c)))   
        d = F.dropout(d, p = 0.35)
        e = F.leaky_relu(self.I5(self.C5(d)))   
        f = F.leaky_relu(self.I6(self.C6(e)))

        f = self.R2(self.R1(f))

        x = F.leaky_relu(self.IT1(self.TC1(f)))
        
        x = torch.cat([x, d], dim = 1)
        x = F.dropout(x, p = 0.35)
        x = F.leaky_relu(self.IT2(self.TC2(x)))

        x = torch.cat([x, c], dim = 1)
        x = F.leaky_relu(self.IT3(self.TC3(x)))

        x = torch.cat([x, b], dim = 1)
        x = F.dropout(x, p = 0.35)
        x = F.leaky_relu(self.IT4(self.TC4(x)))

        x = torch.cat([x, a], dim = 1)
        x = torch.tanh(self.TC5(x))
        return x

class Star_Generator(nn.Module):
    def __init__(self, img_size, in_channels = 3, out_channels = 3, cond_length = 10):
        super(Star_Generator, self).__init__()
        self.cond_length = cond_length
        print(self.cond_length)
        self.h, self.w = img_size[0], img_size[1]
        self.condProj = nn.Linear(cond_length, cond_length*self.h*self.w)

        self.C1 = nn.Conv2d(in_channels + cond_length, 64, kernel_size = 7, stride = 1, padding = 3)
        self.I1 = nn.InstanceNorm2d(64)

        self.C2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.I2 = nn.InstanceNorm2d(128)

        self.C3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.I3 = nn.InstanceNorm2d(256)

        self.R1 = ResBlock(256 + cond_length, specnorm = False, batchnorm = True, use_instance_norm = True)
        self.R2 = ResBlock(256 + cond_length*2, specnorm = False, batchnorm = True, use_instance_norm = True)
        self.R3 = ResBlock(256 + cond_length*3, specnorm = False, batchnorm = True, use_instance_norm = True)
        self.R4 = ResBlock(256 + cond_length*4, specnorm = False, batchnorm = True, use_instance_norm = True)

        self.ResList = nn.ModuleList([self.R1, self.R2, self.R3, self.R4])

        self.TC1 = nn.ConvTranspose2d(256 + cond_length*4, 128, kernel_size = 4, stride = 2, padding = 1)
        self.IT1 = nn.InstanceNorm2d(128)

        self.TC2 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.IT2 = nn.InstanceNorm2d(64)

        self.TC3 = nn.ConvTranspose2d(64, out_channels, kernel_size = 7, stride = 1, padding = 3)        

    def forward(self, x, c):
        #c = c.view(-1, self.cond_length, 1, 1)
        #c = c.repeat(1, 1, x.shape[2], x.shape[3])
        c = F.leaky_relu(self.condProj(c))
        c = c.view(-1, self.cond_length, self.h, self.w)
        x = torch.cat([x, c], dim = 1)

        x = F.leaky_relu(self.I1(self.C1(x)))
        x = F.leaky_relu(self.I2(self.C2(x)))
        x = F.leaky_relu(self.I3(self.C3(x)))

        c = F.interpolate(c, scale_factor = 1/4, align_corners = False, mode = "bilinear")

        for R in self.ResList:
          x = torch.cat([x, c], dim = 1)
          x = R(x)

        x = F.leaky_relu(self.IT1(self.TC1(x)))
        x = F.leaky_relu(self.IT2(self.TC2(x)))
        x = torch.tanh(self.TC3(x))
        return x

class Basic_Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Basic_Discriminator, self).__init__()
        self.C1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)

        self.C2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.B2 = nn.BatchNorm2d(64)
        
        self.C3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.B3 = nn.BatchNorm2d(128)
        
        self.C4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.B4 = nn.BatchNorm2d(256)

        self.C5 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.B5 = nn.BatchNorm2d(512)
        
        self.D1 = nn.Linear(512*4*4, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.C1(x))
        x = F.leaky_relu(self.B2(self.C2(x)))
        x = F.leaky_relu(self.B3(self.C3(x)))
        x = F.leaky_relu(self.B4(self.C4(x)))
        x = F.leaky_relu(self.B5(self.C5(x)))
        x = x.view(-1, 512*4*4)
        x = self.D1(x)
        return x

class Patch_Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Patch_Discriminator, self).__init__()
        self.C1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)

        self.C2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.B2 = nn.BatchNorm2d(64)
        
        self.C3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.B3 = nn.BatchNorm2d(128)
        
        self.C4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.B4 = nn.BatchNorm2d(256)

        self.C5 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.B5 = nn.BatchNorm2d(512)
        
        self.C6 = nn.Conv2d(512, 30, kernel_size = 4, stride = 2, padding = 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.C1(x))
        x = F.leaky_relu(self.B2(self.C2(x)))
        x = F.leaky_relu(self.B3(self.C3(x)))
        x = F.leaky_relu(self.B4(self.C4(x)))
        x = F.leaky_relu(self.B5(self.C5(x)))
        x = F.leaky_relu(self.C6(x))
        return x

class Star_Patch_Discriminator(nn.Module):
    def __init__(self, img_size, in_channels = 3, cond_length = 10):
        super(Star_Patch_Discriminator, self).__init__()
        self.cond_length = cond_length
        self.h, self.w = img_size[0]//64, img_size[1]//64

        self.C1 = nn.Conv2d(in_channels, 32, kernel_size = 4, stride = 2, padding = 1)

        self.C2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.B2 = nn.BatchNorm2d(64)
        
        self.C3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.B3 = nn.BatchNorm2d(128)
        
        self.C4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.B4 = nn.BatchNorm2d(256)

        self.C5 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.B5 = nn.BatchNorm2d(512)

        self.C6 = nn.Conv2d(512, 512, kernel_size = 4, stride = 2, padding = 1)
        self.B6 = nn.BatchNorm2d(512)

        self.C7 = nn.Conv2d(512, 30, kernel_size = 4, stride = 2, padding = 1)

        self.D1 = nn.Linear(512*self.h*self.w, cond_length)

    def forward(self, x):
        x = F.leaky_relu(self.C1(x))
        x = F.leaky_relu(self.B2(self.C2(x)))
        x = F.leaky_relu(self.B3(self.C3(x)))
        x = F.leaky_relu(self.B4(self.C4(x)))
        x = F.leaky_relu(self.B5(self.C5(x)))
        x = F.leaky_relu(self.B6(self.C6(x)))

        h = x.view(-1, 512*self.h*self.w)

        c = self.D1(h)
        x = self.C7(x)
        return x, c