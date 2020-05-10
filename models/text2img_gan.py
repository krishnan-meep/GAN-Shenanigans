import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.building_blocks import SpectralNorm, Self_Attn, ModulatingNet, ResBlock

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, specnorm = True):
		super(ConvBlock, self).__init__()
		self.C1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
		#self.C1 = ResBlock(in_channels, specnorm = specnorm)
		self.C2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

		if specnorm:
			nn.init.kaiming_normal_(self.C1.weight)
			nn.init.kaiming_normal_(self.C2.weight)
			self.C1 = SpectralNorm(self.C1)
			self.C2 = SpectralNorm(self.C2)

	def forward(self, x):
		x = F.leaky_relu(self.C1(x))
		#x = self.C1(x)
		return F.leaky_relu(self.C2(x))

class Text_Embedder(nn.Module):
	def __init__(self, in_size = 50):
		super(Text_Embedder, self).__init__()
		self.in_size = in_size
		self.L = nn.LSTM(in_size, in_size)

	def forward(self, x):
		s = (torch.zeros(1, x.size(0), self.in_size).cuda(), torch.zeros(1, x.size(0), self.in_size).cuda())
		x = x.permute(1, 0 ,2)

		o, s = self.L(x, s)
  
		x = s[0].squeeze(0)

		return x

class Text_Generator(nn.Module):
	def __init__(self, image_size, noise_dim = 128, emb_size = 50, specnorm = True):
		super(Text_Generator, self).__init__()
		self.h, self.w = image_size[0]//16, image_size[1]//16
		self.noise_dim = noise_dim

		self.G_i = SpectralNorm(nn.Linear(emb_size, emb_size))
		self.B_i = SpectralNorm(nn.Linear(emb_size, emb_size))

		self.Const_Start = nn.Parameter(torch.ones(1, 256, self.h, self.w))

		#self.Proj = nn.Linear(noise_dim + emb_size, 256*self.h*self.w)
		self.RGB_init = nn.Conv2d(256, 3, kernel_size = 1)
		self.C_init = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
		self.MN_init = ModulatingNet(256, noise_dim = noise_dim + emb_size, specnorm = specnorm)

		self.Attn = Self_Attn(64, specnorm = specnorm)

		self.layers, self.rgbs, self.mns = [], [], []

		f = 256
		for i in range(4):
			self.layers.append(ConvBlock(f, f//2, specnorm = specnorm))
			self.mns.append(ModulatingNet(f//2, noise_dim = noise_dim + emb_size, specnorm = specnorm))
			RGB = nn.Conv2d(f//2, 3, kernel_size = 1)
			if specnorm: RGB = SpectralNorm(RGB)

			self.rgbs.append(RGB)

			f = f//2

		self.layers, self.rgbs, self.mns = nn.ModuleList(self.layers), nn.ModuleList(self.rgbs), nn.ModuleList(self.mns)

	def forward(self, z, e, t_emb):
		#g_i, b_i = self.G_i(t_emb), self.B_i(t_emb)
		#e = g_i*e + b_i
		z = torch.cat([z, t_emb], dim = 1)

		#x = F.leaky_relu(self.Proj(z)).view(-1, 256, self.h, self.w)
		x = F.leaky_relu(self.C_init(self.Const_Start.repeat(z.size(0), 1, 1, 1)))
		g, b = self.MN_init(z)
		mu, sig = x.mean(dim = 1, keepdim = True), x.std(dim = 1, keepdim = True)
		x = g*(x - mu)/(sig + 1e-14) + b
		o = [torch.tanh(self.RGB_init(x))]

		for i in range(4):
			x = F.interpolate(x, scale_factor = 2)
			x = self.layers[i](x)
			g, b =  self.mns[i](z)
			mu, sig = x.mean(dim = 1, keepdim = True), x.std(dim = 1, keepdim = True)
			x = g*(x - mu)/(sig + 1e-14) + b

			#if i == 2:
 						#x, _ = self.Attn(x)

			t = torch.tanh(self.rgbs[i](x))
			o.append(t)

		return o



class Discriminator(nn.Module):
	def __init__(self, in_channels = 3, specnorm = True):
		super(Discriminator, self).__init__()
		self.layers, self.rgbs, self.mns = [], [], []

		f = 64
		self.C_init = nn.Conv2d(3 + 50, f, kernel_size = 3, padding = 1)
		self.Pool = nn.AvgPool2d(3, stride = 2, padding = 1)

		for i in range(4):
			RGB = nn.Conv2d(3, f, kernel_size = 1)
			if specnorm: RGB = SpectralNorm(RGB)
			self.rgbs.append(RGB)

			self.layers.append(ConvBlock(2*f, 2*f, specnorm))
			self.mns.append(ModulatingNet(2*f, noise_dim = 50, specnorm = specnorm))

			f = f*2

		self.F = nn.Conv2d(f, 1, kernel_size = 1)
		self.layers, self.rgbs, self.mns = nn.ModuleList(self.layers), nn.ModuleList(self.rgbs), nn.ModuleList(self.mns)

		if specnorm:
			self.C_init = SpectralNorm(self.C_init)
			self.F = SpectralNorm(self.F)

	def forward(self, x, t_emb):
		x_i = torch.cat([x[-1], t_emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x[-1].size(2), x[-1].size(3))], dim = 1)
		x_i = F.leaky_relu(self.C_init(x_i))
		x_i = self.Pool(x_i)

		for i in range(4):
			x_rgb = F.leaky_relu(self.rgbs[i](x[-(i+2)]))
			x_c = torch.cat([x_i, x_rgb], dim = 1)

			x_i = self.layers[i](x_c)
			x_i = self.Pool(x_i)

		return self.F(x_i)
