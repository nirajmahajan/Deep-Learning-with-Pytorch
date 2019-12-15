import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

# Define the function to plot the diagram

def plot_decision_regions_3class(model, data_set, show = False):
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
	X = data_set.x.numpy()
	y = data_set.y.numpy()
	h = .02
	x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
	y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1 
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
	_, yhat = torch.max(model(XX), 1)
	yhat = yhat.numpy().reshape(xx.shape)
	plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
	plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label = 'y=0')
	plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label = 'y=1')
	plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label = 'y=2')
	plt.legend()
	if show:
		plt.show()

# Create Data Class
class Data(Dataset):
    
	# Constructor
	def __init__(self, K=3, N=500):
		D = 2
		X = np.zeros((N * K, D)) # data matrix (each row = single example)
		y = np.zeros(N * K, dtype='uint8') # class labels
		for j in range(K):
			ix = range(N * j, N * (j + 1))
			r = np.linspace(0.0, 1, N) # radius
			t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
			X[ix] = np.c_[r * np.sin(t), r*np.cos(t)]
			y[ix] = j
		self.y = torch.from_numpy(y).type(torch.LongTensor)
		self.x = torch.from_numpy(X).type(torch.FloatTensor)
		self.len = y.shape[0]
    
	# Getter
	def __getitem__(self, index):    
		return self.x[index], self.y[index]
    
	# Get Length
	def __len__(self):
		return self.len

	# Plot the diagram
	def plot_stuff(self, show = False):
		plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'o', label="y = 0")
		plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y = 1")
		plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'go', label="y = 2")
		plt.legend()
		if show:
			plt.show()

# Create a neural network class
class Deep_NN(nn.Module):

	# Constructor
	def __init__(self, Layers):
		super(Deep_NN, self).__init__()
		self.hidden = nn.ModuleList()
		for (input_size, output_size) in zip(Layers, Layers[1:]):
			self.hidden.append(nn.Linear(input_size, output_size))

	# Prediction
	def forward(self, x):
		L = len(self.hidden)
		for (l, linear_transform) in zip(range(L), self.hidden):
			if l < L - 1:
				x = torch.relu(linear_transform(x))
			else:
				x = linear_transform(x)
		return x

# The function to calculate the accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()