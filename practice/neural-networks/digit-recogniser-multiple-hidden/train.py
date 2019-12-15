from helpers import *

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')

# Create and print the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# print("Downloaded the training dataset:\n ", train_dataset)
# Create and print the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# print("Downloaded the validating dataset:\n ", validation_dataset)

# Define analysis function
def analyse():
	correct = 0
	incorrect = 0
	for (x,y) in validation_dataset:
		z = model(x.reshape(-1, 28*28))
		_, yhat = torch.max(z, 1)
		if(yhat == y):
			correct += 1
		else:
			incorrect += 1

	return (correct, incorrect, correct/(correct+incorrect))

args = parser.parse_args()

if(not args.use_trained_model):

	# Define a criterion function
	criterion = nn.CrossEntropyLoss()

	# Create Dataloader objects
	trainloader = DataLoader(dataset = train_dataset, batch_size = 2000, shuffle = True)
	validationloader = DataLoader(dataset = validation_dataset, batch_size = 5000, shuffle = False)

	# Define model parameters and create a model
	# The Neural Network will have a single hidden layer with 100 neurons
	in_dim = 784
	out_dim = 10
	Hidden1 = 50
	Hidden2 = 50
	model = Neural_Network(in_dim, Hidden1, Hidden2, out_dim)

	# Define an optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

	# train the model now!! (on 100 epochs)
	epochs = 100
	for epoch in range(epochs):
		print('Running on epoch {}'.format(epoch + 1), flush = True)
		if((epoch+1) % 5 == 0):
			(C, I, A) = analyse()
			print("Accuracy =", A, flush=True)
		for (x,y) in trainloader:
			optimizer.zero_grad()
			z = model(x.view(-1,784))
			loss = criterion(z, y)
			loss.backward()
			optimizer.step()


	with open('model/trained_model.pkl', 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
	if(not os.path.isfile('model/trained_model.pkl')):
		print('Train the model first')
		os._exit(1)

	with open('model/trained_model.pkl', 'rb') as f:
	    model = pickle.load(f)	


# Count the classified and miss classified data using the validation set
(C, I, A) = analyse()

print("Analysis:")
print("Correctly classified data count =", C)
print("Incorrectly classified data count =", I)
print("Accuracy =", A)

plt.show()