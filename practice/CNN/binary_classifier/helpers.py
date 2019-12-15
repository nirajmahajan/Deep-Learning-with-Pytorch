import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
import argparse
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Function to plot channels of Convolution Layer
def plot_channels(W, show = False):
	#number of output channels 
	n_out=W.shape[0]
	#number of input channels 
	n_in=W.shape[1]
	w_min=W.min().item()
	w_max=W.max().item()
	fig, axes = plt.subplots(n_out,n_in)
	fig.subplots_adjust(hspace = 0.1)
	out_index=0
	in_index=0
	#plot outputs as rows inputs as columns 
	for ax in axes.flat:
		if in_index>n_in-1:
			out_index=out_index+1
			in_index=0

		ax.imshow(W[out_index,in_index,:,:], vmin=w_min, vmax=w_max, cmap='seismic')
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		in_index=in_index+1

	if show:
		plt.show()

# Function to show data samples
def show_data(dataset,sample, show = False):
    plt.imshow(dataset.x[sample,0,:,:].numpy(),cmap='gray')
    plt.title('y='+str(dataset.y[sample].item()))
    if show:
    	plt.show()

# Create a Data generator
class Data(Dataset):

	# Constructor
	def __init__(self,N_images=100,offset=0,p=0.9, train=False):
		if train==True:
			np.random.seed(1)  

		#make images multiple of 3 
		N_images=2*(N_images//2)
		images=np.zeros((N_images,1,11,11))
		start1=3
		start2=1
		self.y=torch.zeros(N_images).type(torch.long)

		for n in range(N_images):
			if offset>0:
        
				low=int(np.random.randint(low=start1, high=start1+offset, size=1))
				high=int(np.random.randint(low=start2, high=start2+offset, size=1))
			else:
				low=4
				high=1

			if n<=N_images//2:
				self.y[n]=0
				images[n,0,high:high+9,low:low+3]= np.random.binomial(1, p, (9,3))
			elif  n>N_images//2:
				self.y[n]=1
				images[n,0,low:low+3,high:high+9] = np.random.binomial(1, p, (3,9))

		self.x=torch.from_numpy(images).type(torch.FloatTensor)
		self.len=self.x.shape[0]
		del(images)
		np.random.seed(0)

	def __getitem__(self,index):      
		return self.x[index],self.y[index]
	def __len__(self):
		return self.len


def plot_activations(A,number_rows= 1,name=""):
	A=A[0,:,:,:].detach().numpy()
	n_activations=A.shape[0]

	print(n_activations)
	A_min=A.min().item()
	A_max=A.max().item()

	if n_activations==1:
		# Plot the image.
		plt.imshow(A[0,:], vmin=A_min, vmax=A_max, cmap='seismic')

	else:
		fig, axes = plt.subplots(number_rows, n_activations//number_rows)
		fig.subplots_adjust(hspace = 0.4)
		for i,ax in enumerate(axes.flat):
			if i< n_activations:
				# Set the label for the sub-plot.
				ax.set_xlabel( "activation:{0}".format(i+1))

				# Plot the image.
				ax.imshow(A[i,:], vmin=A_min, vmax=A_max, cmap='seismic')
				ax.set_xticks([])
				ax.set_yticks([])
	plt.show()

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
	from math import floor
	if type(kernel_size) is not tuple:
		kernel_size = (kernel_size, kernel_size)
	h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
	w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
	return h, w

class CNN(nn.Module):

	# Constructor
	def __init__(self, out_1 = 2, out_2 = 1):
		super(CNN, self).__init__()
		self.cnn1=nn.Conv2d(in_channels=1,out_channels=out_1,kernel_size=2,padding=0)
		self.maxpool1=nn.MaxPool2d(kernel_size=2 ,stride=1)

		self.cnn2=nn.Conv2d(in_channels=out_1,out_channels=out_2,kernel_size=2,stride=1,padding=0)
		self.maxpool2=nn.MaxPool2d(kernel_size=2 ,stride=1)

		self.fc1=nn.Linear(out_2*7*7,2)

	# Prediction
	def forward(self, x):
		x = self.maxpool1(torch.relu(self.cnn1(x)))
		x = self.maxpool2(torch.relu(self.cnn2(x)))
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x
