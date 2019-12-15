import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
import argparse
import os
import pickle
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

def plot_accuracy_loss(training_results): 
	plt.subplot(2, 1, 1)
	plt.plot(training_results['training_loss'], 'r')
	plt.ylabel('loss')
	plt.title('training loss iterations')
	plt.subplot(2, 1, 2)
	plt.plot(training_results['validation_accuracy'])
	plt.ylabel('accuracy')
	plt.xlabel('epochs')   
	plt.show()

# Define a function to plot model parameters
def print_model_parameters(model):
	count = 0
	for ele in model.state_dict():
		count += 1
		if count % 2 != 0:
			print ("The following are the parameters for the layer ", count // 2 + 1)
		if ele.find("bias") != -1:
			print("The size of bias: ", model.state_dict()[ele].size())
		else:
			print("The size of weights: ", model.state_dict()[ele].size())

# Define a function to display data
def show_data(data_sample):
	plt.imshow(data_sample.numpy().reshape(28, 28), cmap='gray')
	plt.show()


# Define a nn class
class Neural_Network(nn.Module):

	# Constructor
	def __init__(self, in_dimensions, Hidden, out_dimensions):
		super(Neural_Network, self).__init__()
		self.linear1 = nn.Linear(in_dimensions, Hidden)
		self.linear2 = nn.Linear(Hidden, out_dimensions)

	# Prediction
	def forward(self, x):
		x = torch.relu(self.linear1(x))
		x = self.linear2(x)
		return x

