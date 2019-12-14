import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

### Generate Dataset 
class Data(Dataset):

	# Constructor
	def __init__(self):
		self.x = torch.arange(-1,1,0.1).view(-1,1)
		self.y = torch.zeros(self.x.shape[0], 1)
		self.y[self.x[:, 0] > 0.2] = 1
		self.len = self.x.shape[0]

	# getter
	def __getitem__(self, index):
		return (self.x[index], self.y[index])

	# Get length
	def __len__(self):
		return self.len

### logistic regression model
class logistic_regression(nn.Module):

	# Constructor
	def __init__(self, in_size):
		super(logistic_regression, self).__init__()
		self.linear = nn.Linear(in_size, 1)

	# Prediction
	def forward(self, x):
		yhat = torch.sigmoid(self.linear(x))
		return yhat

# Create dataset
dataset = Data()

# Create model
model = logistic_regression(1)
model.state_dict() ['linear.weight'].data[0] = torch.tensor([[-5]])
model.state_dict() ['linear.bias'].data[0] = torch.tensor([[-10]])

# Create a dataloader
trainloader = DataLoader(dataset = dataset, batch_size = 1)
# Create a loss funtion
loss = nn.BCELoss()
# Create an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 2)

# Train the model
# train model
iter = 100
for epoch in range(iter):
	if(epoch%10 == 0):
		w_found = model.state_dict()['linear.weight']
		b_found = model.state_dict()['linear.bias']
		trained = dataset.x*w_found + b_found
		plt.figure(epoch+1)
		plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label = 'noised')
		plt.plot(dataset.x.numpy(), trained.numpy(),'b', label = 'Epoch:{}'.format(epoch))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Epoch:{}'.format(epoch))
		plt.legend()
	for (x,y) in trainloader:
		yhat = model(x)
		lossv = loss(yhat, y)
		optimizer.zero_grad()
		lossv.backward()
		# print('Loss in the {}th epoch : {}'.format(epoch, lossv))

		optimizer.step()

w_found = model.state_dict()['linear.weight']
b_found = model.state_dict()['linear.bias']
trained = dataset.x*w_found + b_found
plt.figure(iter+1)
plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label = 'data')
plt.plot(dataset.x.numpy(), trained.numpy(),'b', label = 'Classifier')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Results')
plt.legend()
plt.show()
