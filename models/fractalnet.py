import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.building_blocks import FractalBlock, UpBlock, DownBlock, ModulatingNet, MinibatchDiscrimination1d, SpectralNorm

class FractalNet_Generator(nn.Module):
	#Image size should be a tuple, preferably a multiple of 8
	def __init__(self, image_size, level = 0, in_channels = 3, noise_dim = 128, specnorm = False):
		super(FractalNet_Generator, self).__init__()
		self.h, self.w = image_size[0]//8, image_size[1]//8
		self.in_channels = in_channels

		self.Proj = nn.Linear(noise_dim, self.h*self.w*512)
		self.B0 = nn.BatchNorm2d(512)
		self.MN0 = ModulatingNet(512, specnorm = specnorm)

		self.U1 = UpBlock(512, 256, specnorm = specnorm, batchnorm = True)

		self.F1 = FractalBlock(256, level = level, specnorm = specnorm, batchnorm = True)
		self.U2 = UpBlock(256, 128, specnorm = specnorm, batchnorm = True)
		self.MN1 = ModulatingNet(128, specnorm = specnorm)

		self.F2 = FractalBlock(128, level = level, specnorm = specnorm, batchnorm = True)
		self.U3 = UpBlock(128, 64, specnorm = specnorm, batchnorm = True)
		self.MN2 = ModulatingNet(64, specnorm = specnorm)

		self.TC4 = nn.ConvTranspose2d(64, self.in_channels, kernel_size = 3, stride = 1, padding = 1)

		if specnorm:
			self.Proj = SpectralNorm(self.Proj)


	def forward(self, z):
		x = self.Proj(z)
		x = x.view(-1, 512, self.h, self.w)
		x = self.B0(x)
		g, b = self.MN0(z)
		x = g*x + b
		x = F.leaky_relu(x)

		x = self.U1(x)
		x = self.F1(x)
		x = self.U2(x)
		g, b = self.MN1(z)
		x = g*x + b

		x = self.U3(self.F2(x))
		g, b = self.MN2(z)
		x = g*x + b

		x = self.TC4(x)
		x = torch.tanh(x)
		return x

class FractalNet_Discriminator(nn.Module):
	def __init__(self, image_size, level = 0, specnorm = True, mbd_features = None, in_channels = 3):
		super(FractalNet_Discriminator, self).__init__()
		self.h, self.w = image_size[0]//4, image_size[1]//4
		self.in_channels = in_channels
		self.mbd_features = mbd_features

		self.DB1 = DownBlock(self.in_channels, 64, specnorm = specnorm, batchnorm = False)
		self.DB2 = DownBlock(64, 128, specnorm = specnorm, batchnorm = False)
		self.F1 = FractalBlock(128, level = level, specnorm = specnorm, batchnorm = False)
		self.F2 = FractalBlock(128, level = level, specnorm = specnorm, batchnorm = False)

		if mbd_features is not None:
			self.D1 = nn.Linear(128*self.h*self.w + mbd_features, 1)
			self.MBD = MinibatchDiscrimination1d(128*self.h*self.w, mbd_features)
		else:
			self.D1 = nn.Linear(128*self.h*self.w, 1)

		if specnorm:
			self.D1 = SpectralNorm(self.D1)


	def forward(self, x):
		n = torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
		if torch.cuda.is_available():
			n = n.cuda()
		x = x + n*0.001
		x = self.DB1(x)
		x = self.DB2(x)
		x = self.F1(x)
		x = self.F2(x)
		x = x.view(-1, 128*self.h*self.w)

		if self.mbd_features is not None:
			x = self.MBD(x)

		x = self.D1(x)
		return x
