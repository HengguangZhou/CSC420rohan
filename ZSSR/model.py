import torch
import torch.nn as nn
import torch.nn
from math import sqrt
from torch.autograd import Variable

class ZSSRNet(nn.Module):
	def __init__(self, hidden_channel = 64, kernel=3, padding=1, bias=True):
		super(ZSSRNet, self).__init__()
		self.input = nn.Conv2d(in_channels=3, out_channels=hidden_channel, kernel_size=kernel, 
		padding=padding, bias=bias)
		self.conv1 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.conv3 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.conv4 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.conv5 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.conv6 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=kernel,
		padding=padding, bias=bias)
		self.output = nn.Conv2d(in_channels=hidden_channel, out_channels=3, kernel_size=kernel,
		padding=padding, bias=bias)

		self.relu = nn.ReLU(inplace=True)
    
	def forward(self, x):
		
		# residual learning
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		
		out = self.conv1(self.relu(out))
		out = self.conv2(self.relu(out))
		out = self.conv3(self.relu(out))
		out = self.conv4(self.relu(out))
		out = self.conv5(self.relu(out))
		out = self.conv6(self.relu(out))

		out = self.output(self.relu(out))

		out = torch.add(out, residual)	
		return out

