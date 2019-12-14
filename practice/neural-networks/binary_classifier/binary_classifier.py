from helpers import *
torch.manual_seed(1)

# Generate Dataset
dataset = Data()
# Plot Dataset
print("Close the plot window for the code to proceed", flush = True)
PlotStuff(dataset.x, dataset.y, show = True)

# Create a Neural Network with one Hidden Layer with 9 neurons each
model = Network(1, 9, 1)

# Create an optimizer and a Data loader
trainloader = DataLoader(dataset = dataset, batch_size = 100)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
# Create a loss funtion
criterion = nn.BCELoss()

cost = []

epochs = 600
for epoch in range(epochs):
	total = 0
	for x,y in trainloader:
		optimizer.zero_grad()
		yhat = model(x)
		loss = criterion(yhat, y)
		loss.backward()
		optimizer.step()
		total += loss

	if(epoch % 200 == 0):
		plt.figure()
		plt.title('Epoch {}'.format(epoch+1))
		PlotStuff(dataset.x, dataset.y, model, show = False)
	cost.append(total)

plt.figure()
plt.plot(cost)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Cross Entropy Cost')
plt.show()