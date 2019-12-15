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

IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

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
	plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
	plt.title('y = '+ str(data_sample[1]))


# Define a nn class
class CNN(nn.Module):

	# Constructor
	def __init__(self, out_1=16, out_2 = 32, number_of_classes = 10):
		super(CNN, self).__init__()
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
		self.maxpool1=nn.MaxPool2d(kernel_size=2)

		self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
		self.maxpool2=nn.MaxPool2d(kernel_size=2)

		self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)

	# Prediction
	def forward(self, x):
		x = self.maxpool1(torch.relu(self.cnn1(x)))
		x = self.maxpool2(torch.relu(self.cnn2(x)))
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x

