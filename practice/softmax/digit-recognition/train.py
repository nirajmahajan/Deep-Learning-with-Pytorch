from helpers import *

# Create and print the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# print("Downloaded the training dataset:\n ", train_dataset)
# Create and print the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# print("Downloaded the validating dataset:\n ", validation_dataset)


input_dimensions = 28*28
output_dimensions = 10

# Create a model
model = SoftMax(input_dimensions, output_dimensions)

# define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
# Define a loss function
criterion = nn.CrossEntropyLoss()
# Define dataloaders
trainloader = DataLoader(dataset = train_dataset, batch_size = 100)
validationloader = DataLoader(dataset = validation_dataset, batch_size = 5000)

PlotParameters(model)
plt.title('Before Training')


n_epochs = 100
for epoch in range(n_epochs):
	print('Running on epoch {}'.format(epoch), flush = True)
	for x, y in trainloader:
		optimizer.zero_grad()
		z = model(x.view(-1, 28 * 28))
		loss = criterion(z, y)
		loss.backward()
		optimizer.step()
		

PlotParameters(model)
plt.title('After Training')

# Count the classified and miss classified data using the validation set
correct = 0
incorrect = 0
for (x,y) in validation_dataset:
	z = model(x.reshape(-1, 28*28))
	_, yhat = torch.max(z, 1)
	if(yhat == y):
		correct += 1
	else:
		incorrect += 1

print("Analysis:")
print("Correctly classified data count =", correct)
print("Incorrectly classified data count=", incorrect)

plt.show()