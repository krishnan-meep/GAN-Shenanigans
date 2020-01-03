import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.building_blocks import PixelNorm, MinibatchDiscrimination1d, SpectralNorm

#A rewritten version of the code found here########################################
#				https://github.com/odegeasslbc/Progressive-GAN-pytorch
#
#Code is functionally the same
###################################################################################
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)
        
def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class ConvBlock(nn.Module):
	#Just a block of two convolutions with pixelnorm
	def __init__(self, in_channels, out_channels, specnorm = False):
		super(ConvBlock, self).__init__()
		self.C1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
		self.P = PixelNorm()
		self.C2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)

		if specnorm:
			self.C1 = SpectralNorm(self.C1, power_iterations = 3)
			self.C2 = SpectralNorm(self.C2, power_iterations = 3)
		else:
			self.C1 = equal_lr(self.C1)
			self.C2 = equal_lr(self.C2)

	def forward(self, x):
		x = F.leaky_relu(self.P(self.C1(x)))
		x = F.leaky_relu(self.P(self.C2(x)))
		return x


class Prog_Generator(nn.Module):
	def __init__(self, noise_dim = 128, specnorm = False):
		super(Prog_Generator, self).__init__()
		self.noise_dim = noise_dim

		#This guy turns 1x1 into 4x4
		self.Proj = nn.ConvTranspose2d(noise_dim, 128, kernel_size = 4, stride = 1, padding = 0)
		self.Pi = PixelNorm()

		self.C_4x4 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_8x8 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_16x16 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_32x32 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_64x64 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_128x128 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_256x256 = ConvBlock(128, 128, specnorm = specnorm)

		self.To_RGB_8x8 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
		self.To_RGB_16x16 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
		self.To_RGB_32x32 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
		self.To_RGB_64x64 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
		self.To_RGB_128x128 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)
		self.To_RGB_256x256 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)

		self.ConvList = nn.ModuleList([self.C_16x16, self.C_32x32, self.C_64x64, self.C_128x128, self.C_256x256])
		self.To_RGBList = nn.ModuleList([self.To_RGB_8x8, self.To_RGB_16x16, self.To_RGB_32x32, self.To_RGB_64x64,
											self.To_RGB_128x128, self.To_RGB_256x256])

		if specnorm:
			for i in range(len(self.To_RGBList)):
				self.To_RGBList[i] = SpectralNorm(self.To_RGBList[i], power_iterations = 3)
		else:
			for i in range(len(self.To_RGBList)):
				self.To_RGBList[i] = equal_lr(self.To_RGBList[i])



	def forward(self, z, steps = 0, alpha = -1):
		x = z.view(-1, self.noise_dim, 1, 1)

		x = F.leaky_relu(self.Pi(self.Proj(x)))		#Heres our 4x4

		x = self.C_4x4(x)
		x = F.interpolate(x, scale_factor = 2, mode = "bilinear", align_corners = False)
		o = self.C_8x8(x)

		if steps == 1:
			return torch.tanh(self.To_RGB_8x8(o))

		for i in range(0, steps-1):
			o_doub = F.interpolate(o, scale_factor = 2, mode = "bilinear", align_corners = False)
			o_doub = self.ConvList[i](o_doub)

			if i == steps - 2:
				if 0 <= alpha < 1:
					rgb = self.To_RGBList[i](o)		#Previous resolution
					rgb = F.interpolate(rgb, scale_factor = 2, mode = "bilinear", align_corners = False)

					i = self.To_RGBList[i+1](o_doub)	#Current resolution

					masked_out = (1-alpha)*rgb + alpha*i
				else:
					masked_out = self.To_RGBList[i+1](o_doub)
				return torch.tanh(masked_out)

			o = o_doub


class Prog_Discriminator(nn.Module):
	def __init__(self, specnorm = False):
		super(Prog_Discriminator, self).__init__()

		self.C_1 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_2 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_3 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_4 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_5 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_6 = ConvBlock(128, 128, specnorm = specnorm)
		self.C_7 = ConvBlock(128 + 1, 128, specnorm = specnorm)

		self.From_RGB_1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_2 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_3 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_4 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_5 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_6 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
		self.From_RGB_7 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)

		self.D1 = nn.Linear(128*4*4, 1)
		self.total_no_of_layers = 7

		self.ConvList = nn.ModuleList([self.C_1, self.C_2, self.C_3, self.C_4, self.C_5, self.C_6, self.C_7])
		self.From_RGBList = nn.ModuleList([self.From_RGB_1, self.From_RGB_2, self.From_RGB_3, self.From_RGB_4,
										self.From_RGB_5, self.From_RGB_6, self.From_RGB_7])

		if specnorm:
			for i in range(len(self.From_RGBList)):
				self.From_RGBList[i] = SpectralNorm(self.From_RGBList[i], power_iterations = 3)
		else:
			for i in range(len(self.From_RGBList)):
				self.From_RGBList[i] = equal_lr(self.From_RGBList[i])

	def forward(self, x, steps = 0, alpha = -1):

		for i in range(steps, -1, -1):
			#We're building it from the head end, hence this indexing
			layer_index = self.total_no_of_layers - i - 1

			if i == steps:
				o = self.From_RGBList[layer_index](x)

			if i == 0:
				out_std = torch.sqrt(o.var(0, unbiased = False) + 1e-8)
				mean_std = out_std.mean()
				mean_std = mean_std.expand(o.size(0), 1, 4, 4)
				o = torch.cat([o, mean_std], 1)

			o = self.ConvList[layer_index](o)

			if i > 0:
				o = F.interpolate(o, scale_factor = 0.5, mode = "bilinear", align_corners = False)

				if i == steps and 0 <= alpha < 1:
					rgb = F.interpolate(x, scale_factor = 0.5, mode = "bilinear", align_corners = False)
					rgb = self.From_RGBList[layer_index+1](rgb)

					o = (1-alpha)*rgb + alpha*o

		o = o.view(-1, 128*4*4)
		o = self.D1(o)
		return o
