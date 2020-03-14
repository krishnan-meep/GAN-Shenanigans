import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class ResBlock(nn.Module):
  #!!!!!!!!!!!
  #If you set batchnorm to true and use_instance_norm to True, the Resblock does instance_norm
  #instead of batchnorm!!!!
  #!!!!!!!!!!!
  def __init__(self, channels, specnorm = False, batchnorm = False, use_instance_norm = False):

    super(ResBlock, self).__init__()

    #Conv->BN->Relu->Conv->BN->Shortcut->Relu

    self.specnorm = specnorm
    self.batchnorm = batchnorm
    self.C1 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))
    self.C2 = (nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1))    
    if specnorm:
        self.C1 = SpectralNorm(self.C1, power_iterations = 1)
        self.C2 = SpectralNorm(self.C2, power_iterations = 1)

    if use_instance_norm:
      self.B1 = nn.BatchNorm2d(channels)
      self.B2 = nn.BatchNorm2d(channels)
    else:
      self.B1 = nn.InstanceNorm2d(channels)
      self.B2 = nn.InstanceNorm2d(channels)

  def forward(self, x):
    S = x
    if self.batchnorm:
      x = F.relu(self.B1(self.C1(x)))
      x = F.relu(self.B2(self.C2(x)) + S)
    else:
      x = F.relu((self.C1(x)))
      x = F.relu((self.C2(x)) + S)
    return x


class DenseBlock(nn.Module):
  #Produces 160 feature maps
  def __init__(self, channels, specnorm = False, batchnorm = False):
    super(DenseBlock, self).__init__()
    self.specnorm = specnorm
    self.batchnorm = batchnorm

    self.C1 = nn.Conv2d(channels, 32, kernel_size = 1, stride = 1, padding = 0)
    self.C2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
    self.C3 = nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
    self.C4 = nn.Conv2d(96, 32, kernel_size = 3, stride = 1, padding = 1)
    self.C5 = nn.Conv2d(128, 32, kernel_size = 3, stride = 1, padding = 1)

    self.B1 = nn.BatchNorm2d(32)
    self.B2 = nn.BatchNorm2d(32)
    self.B3 = nn.BatchNorm2d(32)
    self.B4 = nn.BatchNorm2d(32)
    self.B5 = nn.BatchNorm2d(32)

    if specnorm:
        self.C1 = SpectralNorm(self.C1, power_iterations = 1)
        self.C2 = SpectralNorm(self.C2, power_iterations = 1)
        self.C3 = SpectralNorm(self.C3, power_iterations = 1)
        self.C4 = SpectralNorm(self.C4, power_iterations = 1)
        self.C5 = SpectralNorm(self.C5, power_iterations = 1)

  def forward(self, x):
    if self.batchnorm:
      a = F.relu(self.B1(self.C1(x)))
      b = F.relu(self.B2(self.C2(a)))
      c = F.relu(self.B3(self.C3(torch.cat([a,b], dim = 1))))
      d = F.relu(self.B4(self.C4(torch.cat([a,b,c], dim = 1))))
      e = F.relu(self.B5(self.C5(torch.cat([a,b,c,d], dim = 1))))
    else:
      a = F.relu((self.C1(x)))
      b = F.relu((self.C2(a)))
      c = F.relu((self.C3(torch.cat([a,b], dim = 1))))
      d = F.relu((self.C4(torch.cat([a,b,c], dim = 1))))
      e = F.relu(self.C5(torch.cat([a,b,c,d], dim = 1)))

    return torch.cat([a,b,c,d,e], dim = 1)

#Only works with Convolutions for now############################################################
#level determines how much you want to split the main convolution path, 0 is just a fractal block
#################################################################################################
class FractalBlock(nn.Module):
  def __init__(self, channels, level = 0, specnorm = False, batchnorm = False):
    super(FractalBlock, self).__init__()
    self.level = level

    if level <= 0:
      self.C1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
      self.C2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
    else:
      self.C1 = FractalBlock(channels, level = level - 1, specnorm = specnorm, batchnorm = batchnorm)
      self.C2 = FractalBlock(channels, level = level - 1, specnorm = specnorm, batchnorm = batchnorm)
    self.C_Side = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    self.B1 = nn.BatchNorm2d(channels)
    self.B2 = nn.BatchNorm2d(channels)
    self.B3 = nn.BatchNorm2d(channels)

    self.batchnorm = batchnorm
    if specnorm:
      if level <= 0:
        self.C1 = SpectralNorm(self.C1)
        self.C2 = SpectralNorm(self.C2)
      self.C_Side = SpectralNorm(self.C_Side)

  def forward(self, x):
    if self.batchnorm:
      if self.level > 0: 
        p1 = self.C1(x)
        p1 = self.C2(p1)
      else:
        p1 = F.leaky_relu(self.B1(self.C1(x)))
        p1 = F.leaky_relu(self.B2(self.C2(p1)))
      p2 = F.leaky_relu(self.B3(self.C_Side(x)))
    else:
      if self.level > 0: 
        p1 = self.C1(x)
        p1 = self.C2(p1)
      else:
        p1 = F.leaky_relu(self.C1(x))
        p1 = F.leaky_relu(self.C2(p1))

      p2 = F.leaky_relu(self.C_Side(x))

    #I assumed this was the element wise mean they meant in the paper, not sure
    x = (p1 + p2)/2
    return x

####################################################################################

class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, specnorm = False, batchnorm = False):
    super(UpBlock, self).__init__()
    self.specnorm = specnorm
    self.batchnorm = batchnorm
    self.U = nn.Upsample(scale_factor = 2, mode = "nearest")
    self.C1 = (nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
    if specnorm:
        self.C1 = SpectralNorm(self.C1, power_iterations= 1)
        self.B1 =  nn.InstanceNorm2d(out_channels)
    else:
        self.B1 =  nn.BatchNorm2d(out_channels)


  def forward(self, x):
    x = self.U(x)
    if self.batchnorm:
      x = F.relu(self.B1(self.C1(x)))
    else:
      x = F.relu((self.C1(x)))
    return x

class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels, specnorm = False, batchnorm = False):
    super(DownBlock, self).__init__()
    self.specnorm = specnorm
    self.batchnorm = batchnorm
    self.C1 = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
    if specnorm:
        self.C1 = SpectralNorm(self.C1, power_iterations= 1)
    self.B1 =  nn.BatchNorm2d(out_channels)

  def forward(self, x):
    if self.batchnorm:
      x = F.relu(self.B1(self.C1(x)))
    else:
      x = F.relu((self.C1(x)))
    return x


#############################################################################
#Based on Self Modulation of GANs - https://arxiv.org/abs/1810.01365
#Try placing this after BatchNorm or replace BatchNorm gamma and beta with 
#these gamma and beta
#############################################################################
class ModulatingNet(nn.Module):
  def __init__(self, out_features, noise_dim = 128, hidden_features = 32, specnorm = False):
    super(ModulatingNet, self).__init__()
    self.L1 = nn.Linear(noise_dim, hidden_features)
    self.out_features = out_features
    self.Gamma = nn.Linear(hidden_features, out_features, bias = False)
    self.Beta = nn.Linear(hidden_features, out_features, bias = False)

    if specnorm:
      self.L1 = SpectralNorm(self.L1)
      self.Gamma = SpectralNorm(self.Gamma)
      self.Beta = SpectralNorm(self.Beta)

  def forward(self, z):
    x = F.relu(self.L1(z))
    return self.Gamma(x).view(-1, self.out_features, 1, 1), self.Beta(x).view(-1, self.out_features, 1, 1)


#############################################################################
#From the torchgan library on github
#Haven't figured out the best value for out_features, needs experimentation
#############################################################################
class MinibatchDiscrimination1d(nn.Module):
  def __init__(self, in_features, out_features, intermediate_features=16):
      super(MinibatchDiscrimination1d, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.intermediate_features = intermediate_features

      self.T = nn.Parameter(
          torch.Tensor(in_features, out_features, intermediate_features)
      )
      nn.init.normal_(self.T)

  def forward(self, x):

      M = torch.mm(x, self.T.view(self.in_features, -1))
      M = M.view(-1, self.out_features, self.intermediate_features).unsqueeze(0)
      M_t = M.permute(1, 0, 2, 3)
      # Broadcasting reduces the matrix subtraction to the form desired in the paper
      out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
      return torch.cat([x, out], 1)



def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

#Spectral normalization code was taken from  ################################
#    https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
#
#You may want to experiment with power_iterations, sometimes I do 3 and it
#seems to help
#############################################################################
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


#Self attention code was taken from  ################################
#    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
#############################################################################
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation = "relu", specnorm = False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        if specnorm:
          self.query_conv = SpectralNorm(self.query_conv)
          self.key_conv = SpectralNorm(self.key_conv)
          self.value_conv = SpectralNorm(self.value_conv)

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

#Gradient penalty code was taken from  ################################
#    https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
#
#Default gp weightage is 10
#############################################################################
def GradientPenalty(discriminator_model, real_data, generated_data, gp_weight = 10):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True)

    if torch.cuda.is_available():
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator_model(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if torch.cuda.is_available() else torch.ones(
                           prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)

###Taken from###
#https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#############
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = torch.nn.functional.l1_loss(x, y)
        return loss