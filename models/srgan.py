import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.building_blocks import ResBlock, ModulatingNet, SpectralNorm, DownBlock

class ResBlock(nn.Module):
  def __init__(self, channels, specnorm = False, batchnorm = False):
    super(ResBlock, self).__init__()
    self.specnorm = specnorm
    self.batchnorm = batchnorm
    self.PRelu = nn.PReLU()
    self.C1 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))
    self.C2 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))
    if specnorm:
        self.C1 = SpectralNorm(self.C1, power_iterations = 1)
        self.C2 = SpectralNorm(self.C2, power_iterations = 1)

    self.B1 = nn.InstanceNorm2d(channels)
    self.B2 = nn.InstanceNorm2d(channels)

  def forward(self, x):
    S = x
    if self.batchnorm:
      x = self.PRelu(self.B1(self.C1(x)))
      x = self.B2(self.C2(x)) + S
    else:
      x = self.PRelu((self.C1(x)))
      x = (self.C2(x)) + S
    return x

class UpBlock(nn.Module):
	def __init__(self, channels, specnorm = True):
		super(UpBlock, self).__init__()

		self.C = nn.Conv2d(channels, channels*4, kernel_size = 3, padding = 1, stride = 1)
		self.PixSh = nn.PixelShuffle(2)
		self.PRelu = nn.PReLU()

		if specnorm:
			self.C = SpectralNorm(self.C)

	def forward(self, x):
		x = self.PixSh(self.C(x))
		return self.PRelu(x)

class SRGenerator(nn.Module):
	def __init__(self, in_channels = 3, out_channels = 3, specnorm = True):
		super(SRGenerator, self).__init__()
		self.in_channels = in_channels
		self.C_i = nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1)
		self.C_m = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
		self.C_f = nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1)
		self.PRelu = nn.PReLU()

		self.R = []
		for i in range(5):
			self.R.append(ResBlock(64, specnorm = specnorm))
		self.R = nn.ModuleList(self.R)

		self.U = []
		for i in range(2):
			self.U.append(UpBlock(64, specnorm = specnorm))
		self.U = nn.ModuleList(self.U)

		if specnorm:
			self.C_i = SpectralNorm(self.C_i)
			self.C_m = SpectralNorm(self.C_m)
			self.C_f = SpectralNorm(self.C_f)

	def forward(self, x):
		x = self.PRelu(self.C_i(x))
		x_i = x

		for block in self.R:
			x = block(x)

		x = self.PRelu(self.C_m(x))
		x = x + x_i

		for block in self.U:
			x = block(x)

		x = torch.tanh(self.C_f(x))
		return x

class SRDiscriminator(nn.Module):
	def __init__(self, in_channels = 3, specnorm = True):
		super(SRDiscriminator, self).__init__()

		f = [3, 64, 64, 128, 128, 256, 256, 512, 512, 1024]
		self.D = []
		for i in range(len(f) - 1):
			self.D.append(DownBlock(f[i], f[i+1], specnorm = specnorm))
		self.D = nn.ModuleList(self.D)

		self.C_f = nn.Conv2d(1024, 1, 1, 1, 1)

		if specnorm:
			self.C_f = SpectralNorm(self.C_f)

	def forward(self, x):
		for block in self.D:
			x = block(x)
		return self.C_f(x)