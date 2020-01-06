import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.building_blocks import ResBlock, UpBlock, DownBlock, ModulatingNet, MinibatchDiscrimination1d, SpectralNorm

class cResNet_Generator(nn.Module):
	#Image size should be a tuple, preferably a multiple of 8
	def __init__(self, image_size, in_channels = 3, noise_dim = 128, emb_dim = 50, specnorm = False):
		super(cResNet_Generator, self).__init__()
		self.h, self.w = image_size[0]//8, image_size[1]//8
		self.in_channels = in_channels
		self.emb_dim = emb_dim

		self.Proj = nn.Linear(noise_dim, self.h*self.w*512)
		self.condProj = nn.Linear(emb_dim, self.h*self.w*emb_dim)
		self.B0 = nn.BatchNorm2d(512)
		self.MN0 = ModulatingNet(512, specnorm = specnorm)

		self.U1 = UpBlock(512 + emb_dim, 256, specnorm = specnorm, batchnorm = True)

		self.R1 = ResBlock(256, specnorm = specnorm, batchnorm = True)
		self.U2 = UpBlock(256, 128, specnorm = specnorm, batchnorm = True)
		self.MN1 = ModulatingNet(128, specnorm = specnorm)

		self.R2 = ResBlock(128, specnorm = specnorm, batchnorm = True)
		self.U3 = UpBlock(128, 64, specnorm = specnorm, batchnorm = True)
		self.MN2 = ModulatingNet(64, specnorm = specnorm)

		self.TC4 = nn.ConvTranspose2d(64, self.in_channels, kernel_size = 3, stride = 1, padding = 1)

		if specnorm:
			self.Proj = SpectralNorm(self.Proj)

	 
	def forward(self, z, c):
		x = self.Proj(z)
		x = x.view(-1, 512, self.h, self.w)
		x = self.B0(x)
		g, b = self.MN0(z)
		x = g*x + b
		x = F.leaky_relu(x)

		c = self.condProj(c)
		c = F.leaky_relu(c)
		c = c.view(-1, self.emb_dim, self.h, self.w)
		x = torch.cat([x, c], dim = 1)

		x = self.U1(x)
		x = self.U2(self.R1(x))
		g, b = self.MN1(z)
		x = g*x + b

		x = self.U3(self.R2(x))
		g, b = self.MN2(z)
		x = g*x + b

		x = self.TC4(x)
		x = torch.tanh(x)
		return x

class cResNet_Discriminator(nn.Module):
	def __init__(self, image_size, emb_dim = 50, specnorm = True, mbd_features = None, in_channels = 3):
		super(cResNet_Discriminator, self).__init__()
		self.h, self.w = image_size[0]//4, image_size[1]//4
		self.in_channels = in_channels
		self.mbd_features = mbd_features
		self.emb_dim = emb_dim

		self.DB1 = DownBlock(self.in_channels, 64, specnorm = specnorm, batchnorm = False)
		self.DB2 = DownBlock(64, 128, specnorm = specnorm, batchnorm = False)
		self.R1 = ResBlock(128, specnorm = specnorm, batchnorm = False)
		self.R2 = ResBlock(128, specnorm = specnorm, batchnorm = False)

		if mbd_features is not None:
			self.D1 = nn.Linear(128*self.h*self.w + mbd_features + emb_dim, 1)
			self.MBD = MinibatchDiscrimination1d(128*self.h*self.w, mbd_features)
		else:
			self.D1 = nn.Linear(128*self.h*self.w + emb_dim, 1)

		if specnorm:
			self.D1 = SpectralNorm(self.D1)


	def forward(self, x, c):
		n = torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
		c_n = torch.randn(c.shape[0], c.shape[1])

		if torch.cuda.is_available():
			n, c_n = n.cuda(), c_n.cuda()

		x = x + n*0.001
		c = c + c_n*0.001
		x = self.DB1(x)
		x = self.DB2(x)
		x = self.R1(x)
		x = self.R2(x)
		x = x.view(-1, 128*self.h*self.w)

		if self.mbd_features is not None:
			x = self.MBD(x)

		x = torch.cat([x, c], dim = 1)
		x = self.D1(x)
		return x
