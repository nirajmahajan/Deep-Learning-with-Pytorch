import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)

# Create Data
class Data(Dataset):
	
	# Constructor
	def __init__(self):
		self.x = torch.arange(-3, 3, 0.1).view(-1,1)
		self.f = 12*self.x - 20
		self.y = self.f + 0.2*torch.randn(self.x.size())
		self.len = self.x.shape[0]

	# Getter
	def __getitem__(self, index):
		return (self.x[index], self.y[index])

	# Get Length
	def __len__(self):
		return self.len

# create a linear regression class
class linear_regression(nn.Module):

	# Constructor
	def __init__(self, input_size, output_size):
		super(linear_regression, self).__init__()
		self.linear = nn.Linear(input_size, output_size)

	# Prediction
	def forward(self, x):
		yhat = self.linear(x)
		return yhat

# create dataset instance
dataset = Data()
# plot the data
# plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label = 'y')
# plt.plot(dataset.x.numpy(), dataset.f.numpy(), label = 'f')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Dataset')
# plt.legend()

### Now for training the data
# Create a loss function
loss = nn.MSELoss()

# Create a model and a SGD optimizer
model = linear_regression(1,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# create a Dataloader object
trainloader = DataLoader(dataset = dataset, batch_size = 1)

# train model
iter = 5
for epoch in range(iter):
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
	# plt.show()
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
plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label = 'noised')
plt.plot(dataset.x.numpy(), trained.numpy(),'b', label = 'Final')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Results')
plt.legend()
plt.show()