from helpers import *

torch.manual_seed(1)

# create Dataset
dataset = Data()
plt.figure()
plt.title('Dataset')
dataset.plot_stuff(show = False)
# dataset.y = dataset.y.view(-1)

# Create a model with 3 hidden layers with 10 neurons
Layers = [2, 10, 10, 10, 3]
model = Deep_NN(Layers)

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Create a dataLoader
trainloader = DataLoader(dataset = dataset, batch_size = 20)

# define a criterion function
criterion = nn.CrossEntropyLoss()

# Train the model
epochs = 1000
for epoch in range(epochs):
	if(epoch % 20 == 0):
		print('Running on epoch {}'.format(epoch + 1), flush = True)

	if(epoch % 50 == 0):
		plt.figure()
		plt.title('Epoch {}'.format(epoch+1))
		plot_decision_regions_3class(model, dataset, show = False)		

	for x,y in trainloader:
		optimizer.zero_grad()
		yhat = model(x)
		loss = criterion(yhat, y)
		loss.backward()
		optimizer.step()

plt.figure()
plt.title('Results')
plot_decision_regions_3class(model, dataset, show = True)