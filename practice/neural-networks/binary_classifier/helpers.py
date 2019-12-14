import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def get_hist(model,data_set):
	activations=model.activation(data_set.x)
	for i,activation in enumerate(activations):
		plt.hist(activation.numpy(),4,density=True)
		plt.title("Activation layer " + str(i+1))
		plt.xlabel("Activation")
		plt.xlabel("Activation")
		plt.legend()
		plt.show()

def PlotStuff(X,Y,model=None,leg=False, show=False):
	plt.plot(X[Y==0].numpy(),Y[Y==0].numpy(),'or',label='training points y=0 ' )
	plt.plot(X[Y==1].numpy(),Y[Y==1].numpy(),'ob',label='training points y=1 ' )

	if model!=None:
		plt.plot(X.numpy(),model(X).detach().numpy(),label='neural network ')

	plt.legend()
	if show:
		plt.show()


### Generate Data
class Data(Dataset):

	# Constructor
	def __init__(self):
		super(Data, self).__init__()
		self.x = torch.linspace(-20, 20, 100).view(-1, 1)
		self.y = torch.zeros(self.x.shape[0])
		self.y[(self.x[:,0]>-10)& (self.x[:,0]<-5)]=1
		self.y[(self.x[:,0]>5)& (self.x[:,0]<10)]=1
		self.y=self.y.view(-1,1)
		self.len=self.x.shape[0]
		
	# Getter
	def __getitem__(self, index):
		return (self.x[index], self.y[index])

	# lenth function
	def __len__(self):
		return self.len

### Neural Network class
class Network(nn.Module):

	# Constructor
	def __init__(self, in_size, H, out_size):
		super(Network, self).__init__()
		self.linear1 = nn.Linear(in_size, H)
		self.linear2 = nn.Linear(H, out_size)

	# Prediction
	def forward(self, x):
		x = torch.sigmoid(self.linear1(x))
		x = torch.sigmoid(self.linear2(x))
		return x