import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.building_blocks import SpectralNorm, Self_Attn

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, specnorm = True):
		super(ConvBlock, self).__init__()
		self.C1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
		self.C2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

		if specnorm:
			self.C1 = SpectralNorm(self.C1)
			self.C2 = SpectralNorm(self.C2)

	def forward(self, x):
		x = F.leaky_relu(self.C1(x))
		return F.leaky_relu(self.C2(x))

class Generator(nn.Module):
	def __init__(self, image_size, noise_dim = 128, specnorm = True):
		super(Generator, self).__init__()
		self.h, self.w = image_size[0]//8, image_size[1]//8

		self.Proj = nn.Linear(128, 128*self.h*self.w)
		self.RGB_init = nn.Conv2d(128, 3, kernel_size = 1)

		self.Attn = Self_Attn(32, specnorm = specnorm)

		self.layers, self.rgbs = [], []

		f = 128
		for i in range(3):
			self.layers.append(ConvBlock(f, f//2, specnorm = specnorm))

			RGB = nn.Conv2d(f//2, 3, kernel_size = 1)
			if specnorm: RGB = SpectralNorm(RGB)

			self.rgbs.append(RGB)

			f = f//2

		self.layers, self.rgbs = nn.ModuleList(self.layers), nn.ModuleList(self.rgbs)

	def forward(self, z):
		x = F.leaky_relu(self.Proj(z)).view(-1, 128, self.h, self.w)
		o = [torch.tanh(self.RGB_init(x))]

		for i in range(3):
			x = F.interpolate(x, scale_factor = 2)
			x = self.layers[i](x)

			if i == 1:
 						x, _ = self.Attn(x)

			t = torch.tanh(self.rgbs[i](x))
			o.append(t)

		return o


class Discriminator(nn.Module):
	def __init__(self, in_channels = 3, specnorm = True):
		super(Discriminator, self).__init__()
		self.layers, self.rgbs = [], []

		f = 32
		self.C_init = nn.Conv2d(3, f, kernel_size = 3, padding = 1)
		self.Pool = nn.AvgPool2d(3, stride = 2, padding = 1)

		for i in range(3):
			RGB = nn.Conv2d(3, f, kernel_size = 1)
			if specnorm: RGB = SpectralNorm(RGB)
			self.rgbs.append(RGB)

			self.layers.append(ConvBlock(2*f, 2*f, specnorm))
			f = f*2

		self.F = nn.Conv2d(f, 1, kernel_size = 1)
		self.layers, self.rgbs = nn.ModuleList(self.layers), nn.ModuleList(self.rgbs)

		if specnorm:
			self.C_init = SpectralNorm(self.C_init)
			self.F = SpectralNorm(self.F)

	def forward(self, x):
		x_i = F.leaky_relu(self.C_init(x[-1]))
		x_i = self.Pool(x_i)

		for i in range(3):
			x_rgb = torch.tanh(self.rgbs[i](x[-(i+2)]))
			x_c = torch.cat([x_i, x_rgb], dim = 1)

			x_i = self.layers[i](x_c)
			x_i = self.Pool(x_i)

		return self.F(x_i)
