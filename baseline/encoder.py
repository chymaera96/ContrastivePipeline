import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

d = 128
h = 1024
u = 32
v = int(h/d)
chang_fp = [d,d,2*d,2*d,4*d,4*d,h,h]

class Encoder(nn.Module):
	def __init__(self, in_channels=1, stride=2, kernel_size=3):
		super(Encoder, self).__init__()
		self.in_channels = in_channels
		self.stride = stride
		self.kernel_size = kernel_size
		self.conv_layers = self.create_conv_layers(chang_fp)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0],-1)
		return x


	def create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels
		kernel_size = self.kernel_size
		stride = self.stride
		shape = [1,256,32]
		for channels in architecture:

			layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,1)))
			shape[0] = channels
			shape[2] = int(np.ceil(shape[2]/2))
			layers.append(nn.LayerNorm(shape))
			layers.append(nn.ReLU())
			layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(kernel_size,1), stride=(stride,1), padding=(1,0)))
			shape[1] = int(np.ceil(shape[1]/2))
			layers.append(nn.LayerNorm(shape))
			layers.append(nn.ReLU())

			in_channels = channels

		return nn.Sequential(*layers)
	

class Residual(nn.Module):
    def __init__(self, inplanes, channels, temporal_conv, strides, downsample = None):
        super(Residual, self).__init__()

        if temporal_conv:
            kernels = [[3,1],[1,3],[1,1]]
        else:
            kernels = [[1,1],[1,3],[1,1]]

        self.conv1 = nn.Sequential(
                        nn.Conv2d(inplanes, channels[1], kernel_size = kernels[0], stride = [1, strides[0]], padding = [int(kernels[0][0] / 2) ,0]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[1], kernel_size = kernels[1], stride = [1, strides[1]], padding = [0, 1]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[2], kernel_size = kernels[2], stride = [1,strides[2]], padding = 0),
                        nn.GroupNorm(channels[2], channels[2]))
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # print(f" Downsample = {self.downsample}")
        # print(f"Inside res unit: {x.shape}")
        residual = x
        out = self.conv1(x)
        # print(f"After conv1 {out.shape}")
        out = self.conv2(out)
        # print(f"After conv2 {out.shape}")
        out = self.conv3(out)
        # print(f"After conv3 {out.shape}")
        if self.downsample:
            residual = self.downsample(x)
        # print(f"Residual shape {residual.shape}")
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layers = cfg['layers']
        self.channels = [[64,128,128],[128,128,256],[256,256,512],[512,512,1024]] # [dim_in, dim_inner, dim_out]
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = [1,7], stride = 2, padding = [0,3]), #correct
                        nn.GroupNorm(64,64),
                        nn.ReLU())
        
        self.layer2 = self._make_layer(block, self.channels[0], self.layers[0], temporal_conv=False)
        self.layer3 = self._make_layer(block, self.channels[1], self.layers[1], temporal_conv=False)
        self.layer4 = self._make_layer(block, self.channels[2], self.layers[2], temporal_conv=True)
        self.layer5 = self._make_layer(block, self.channels[3], self.layers[3], temporal_conv=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, channels, blocks, temporal_conv, type=None, strided=True):
        downsample = None
        layers = []
        inplanes = channels[0]

        strides = [1,1,1]
        if strided:
            strides[0] = 2

        for i in range(blocks):

            if strides[0] != 1 or inplanes != channels[-1]:

                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, channels[-1], kernel_size=1, stride=[1,strides[0]]),
                    nn.BatchNorm2d(channels[-1]),
                )
            else:
                downsample=None

            layers.append(block(inplanes, channels, temporal_conv, strides, downsample))
            inplanes = channels[-1]
            strides = [1,1,1]

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.layer2(x)
        # print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        # print(f"After layer3: {x.shape}")
        x = self.layer4(x)
        # print(f"After layer4: {x.shape}")
        x = self.layer5(x)
        # print(f"After layer5: {x.shape}")
        x = self.avgpool(x)
        # print(f"After avgpool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")

        return x