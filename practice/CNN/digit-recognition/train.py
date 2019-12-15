from helpers import *

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')

# Create and print the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
# print("Downloaded the training dataset:\n ", train_dataset)
# Create and print the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)
# print("Downloaded the validating dataset:\n ", validation_dataset)

# Create Dataloader objects
trainloader = DataLoader(dataset = train_dataset, batch_size = 100)
validationloader = DataLoader(dataset = validation_dataset, batch_size = 5000)

args = parser.parse_args()

if(not args.use_trained_model):

	# Define a criterion function
	criterion = nn.CrossEntropyLoss()

	# Define model parameters and create a model
	# The Neural Network will have a single hidden layer with 100 neurons
	in_dim = 784
	out_dim = 10
	Hidden = 100
	model = CNN(out_1=16, out_2=32)

	# Define an optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

	# Define analysis function
	def analyse():
		correct=0
		N_test=len(validation_dataset)
		#perform a prediction on the validation  data  
		for x_test, y_test in validationloader:
			model.eval()
			z = model(x_test)
			_, yhat = torch.max(z.data, 1)
			correct += (yhat == y_test).sum().item()
		accuracy = correct / N_test
		return (correct, N_test-correct, accuracy)

	# train the model now!! (on 100 epochs)
	epochs = 10
	for epoch in range(epochs):
		print('Ran {} epochs till now'.format(epoch), flush = True)
		(C, I, A) = analyse()
		print("Accuracy =", A, flush=True)
		for (x,y) in trainloader:
			model.train()
			optimizer.zero_grad()
			z = model(x)
			loss = criterion(z, y)
			loss.backward()
			optimizer.step()


	with open('model/trained_model.pkl', 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
	# Define analysis function
	def analyse():
		correct=0
		N_test=len(validation_dataset)
		#perform a prediction on the validation  data  
		for x_test, y_test in validationloader:
			model.eval()
			z = model(x_test)
			_, yhat = torch.max(z.data, 1)
			correct += (yhat == y_test).sum().item()
		accuracy = correct / N_test
		return (correct, N_test-correct, accuracy)

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