import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from building_blocks import ResBlock, UpBlock, DownBlock, ModulatingNet, MinibatchDiscrimination1d, SpectralNorm

###########################################
#Generator Network#########################
class Generator(nn.Module):
	#Image size should be a tuple, preferably a multiple of 8
	def __init__(self, image_size, noise_dim = 128, specnorm = False):
		super(Generator, self).__init__()
		self.h, self.w = image_size[0]//8, image_size[1]//8

		self.Proj = nn.Linear(noise_dim, self.h*self.w*512)
		self.B0 = nn.BatchNorm2d(512)
		self.MN0 = ModulatingNet(512, specnorm = specnorm)

		self.TC1 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
		self.B1 = nn.BatchNorm2d(256)
		self.MN1 = ModulatingNet(256, specnorm = specnorm)

		self.TC2 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
		self.B2 = nn.BatchNorm2d(128)
		self.MN2 = ModulatingNet(128, specnorm = specnorm)

		self.TC3 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
		self.B3 = nn.BatchNorm2d(64)
		self.MN3 = ModulatingNet(64, specnorm = specnorm)

		self.TC4 = nn.ConvTranspose2d(64, 3, kernel_size = 3, stride = 1, padding = 1)

		if specnorm:
			self.Proj = SpectralNorm(self.Proj)
			self.TC1 = SpectralNorm(self.TC1)
			self.TC2 = SpectralNorm(self.TC2)
			self.TC3 = SpectralNorm(self.TC3)
			self.TC4 = SpectralNorm(self.TC4)
	 
	def forward(self, z):
		x = self.Proj(z)
		x = x.view(-1, 512, self.h, self.w)
		x = self.B0(x)
		g, b = self.MN0(z)
		x = g*x + b
		x = F.leaky_relu(x)

		x = self.TC1(x)
		x = self.B1(x)
		g, b = self.MN1(z)
		x = g*x + b
		x = F.leaky_relu(x)
	
		x = self.TC2(x)
		x = self.B2(x)
		g, b = self.MN2(z)
		x = g*x + b
		x = F.leaky_relu(x)

		x = self.TC3(x)
		x = self.B3(x)
		g, b = self.MN3(z)
		x = g*x + b
		x = F.leaky_relu(x)
	
		x = self.TC4(x)
		x = torch.tanh(x)
		return x

##########################################
#Discriminator Network####################
class Discriminator(nn.Module):
	def __init__(self, image_size):
		super(Discriminator, self).__init__()
		self.h, self.w = image_size[0]//16, image_size[1]//16

		self.DB1 = DownBlock(3, 64, specnorm = True, batchnorm = False)
		self.DB2 = DownBlock(64, 128, specnorm = True, batchnorm = False)
		self.DB3 = DownBlock(128, 256, specnorm = True, batchnorm = False)
		self.DB4 = DownBlock(256, 512, specnorm = True, batchnorm = False)

		self.D1 = SpectralNorm(nn.Linear(512*self.h*self.w, 1))
		self.MBD = MinibatchDiscrimination1d(512*self.h*self.w, 512)

	def forward(self, x):
		n = torch.randn(x.shape[0], x.shape[1], x.shape[2], 1)
		if torch.cuda.is_available():
			n = n.cuda()
		x = x + n*0.01
		x = self.DB1(x)
		x = self.DB2(x)
		x = self.DB3(x)
		x = self.DB4(x)
		x = x.view(-1, 512*self.h*self.w)
		#x = self.MBD(x)
		x = self.D1(x)
		return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise_dim = 128

netG = Generator(image_size = (32,32), noise_dim = 128).to(device)