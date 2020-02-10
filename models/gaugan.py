import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.building_blocks import ResBlock, UpBlock, DownBlock, ModulatingNet, MinibatchDiscrimination1d, SpectralNorm

class SPADE(nn.Module):
	def __init__(self, out_channels, in_channels = 3, specnorm = False):
		super(SPADE, self).__init__()

		self.C_P = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
		self.G = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
		self.B = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

		if specnorm:
			self.C_P = SpectralNorm(self.C_P, power_iterations = 1)
			self.G = SpectralNorm(self.G, power_iterations = 1)
			self.B = SpectralNorm(self.B, power_iterations = 1)

	def forward(self, x, s):
		s = F.leaky_relu(self.C_P(s))
		scale_factor = x.size(2)/s.size(2)
		s = F.interpolate(s, scale_factor = scale_factor, mode = "nearest")

		gamma = self.G(s)
		beta = self.G(s)

		return x*gamma + beta

class SPADE_Block(nn.Module):
	def __init__(self, channels, seg_channels, specnorm = False):
		super(SPADE_Block, self).__init__()
		#Conv->BN->Relu->Conv->BN->Shortcut->Relu

		self.specnorm = specnorm
		self.C1 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))
		self.C2 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))    
		if specnorm:
			self.C1 = SpectralNorm(self.C1, power_iterations = 1)
			self.C2 = SpectralNorm(self.C2, power_iterations = 1)

		self.S1 = SPADE(out_channels = channels, in_channels = seg_channels, specnorm = specnorm)
		self.S2 = SPADE(out_channels = channels, in_channels = seg_channels, specnorm = specnorm)

	def forward(self, x, s):
		x_i = x
		x = F.relu(self.S1(x, s))
		x = F.relu(self.S2(self.C1(x), s))
		x = x_i + self.C2(x)
		return x

class SPADE_Generator(nn.Module):
	#Image size should be a tuple, preferably a multiple of 8
	def __init__(self, image_size, in_channels = 3, noise_dim = 128, seg_channels = 3, specnorm = False):
		super(SPADE_Generator, self).__init__()
		self.h, self.w = image_size[0]//16, image_size[1]//16
		self.in_channels = in_channels

		self.Proj = nn.Linear(noise_dim, self.h*self.w*512)

		self.U1 = UpBlock(512, 256, specnorm = specnorm, batchnorm = True)

		self.R1 = SPADE_Block(256, seg_channels = seg_channels, specnorm = specnorm)
		self.U2 = UpBlock(256, 128, specnorm = specnorm, batchnorm = True)

		self.R2 = SPADE_Block(128, seg_channels = seg_channels, specnorm = specnorm)
		self.U3 = UpBlock(128, 128, specnorm = specnorm, batchnorm = True)

		self.R3 = SPADE_Block(128, seg_channels = seg_channels, specnorm = specnorm)
		self.U4 = UpBlock(128, 64, specnorm = specnorm, batchnorm = True)

		self.TC4 = nn.ConvTranspose2d(64, self.in_channels, kernel_size = 3, stride = 1, padding = 1)

		if specnorm:
			self.Proj = SpectralNorm(self.Proj)

	def forward(self, z, s):
		x = self.Proj(z)
		x = x.view(-1, 512, self.h, self.w)
		x = F.leaky_relu(x)

		x = self.U1(x)
		x = self.U2(self.R1(x, s))
		x = self.U3(self.R2(x, s))
		x = self.U4(self.R3(x, s))

		x = self.TC4(x)
		x = torch.tanh(x)
		return x

class Patch_Discriminator(nn.Module):
	def __init__(self, image_size, specnorm = True, in_channels = 3):
		super(Patch_Discriminator, self).__init__()
		self.h, self.w = image_size[0]//16, image_size[1]//16
		self.in_channels = in_channels

		self.DB1 = DownBlock(self.in_channels, 64, specnorm = specnorm, batchnorm = False)
		self.DB2 = DownBlock(64, 128, specnorm = specnorm, batchnorm = False)
		self.DB3 = DownBlock(128, 128, specnorm = specnorm, batchnorm = False)
		self.DB4 = DownBlock(128, 128, specnorm = specnorm, batchnorm = False)
		self.R1 = ResBlock(128, specnorm = specnorm, batchnorm = False)
		#self.R2 = ResBlock(128, specnorm = specnorm, batchnorm = False)
		self.F = nn.Conv2d(128, 1, kernel_size = 3, padding = 1)

		if specnorm:
				self.F = SpectralNorm(self.F)


	def forward(self, x):
		n = torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
		if torch.cuda.is_available():
			n = n.cuda()
		x = x + n*0.001
		x = self.DB1(x)
		x = self.DB2(x)
		x = self.DB3(x)
		x = self.DB4(x)
		x = self.R1(x)
		#x = self.R2(x)
		x = self.F(x)

		return x