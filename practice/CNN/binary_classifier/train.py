from helpers import *

torch.manual_seed(4)

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')

# Generate Data
train_dataset = Data(N_images = 10000)
validation_dataset = Data(N_images = 1000, train = False)

args = parser.parse_args()

if(not args.use_trained_model):

	# Create model
	model=CNN(2,1)

	# Define a criterion function
	criterion=nn.CrossEntropyLoss()
	# Define an optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# Define dataloaders
	train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=10)
	validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=20)

	# Train the data
	N_test=len(validation_dataset)
	cost_list=[]
	accuracy_list=[]
	cost=0
	epochs = 10
	for epoch in range(epochs):
		cost = 0
		print('Running on the {}th epoch'.format(epoch+1))
		for x,y in train_loader:
			optimizer.zero_grad()
			z = model(x)
			loss = criterion(z, y)
			loss.backward()
			optimizer.step()
			cost += loss
		cost_list.append(cost)

		correct = 0
		for x_test, y_test in validation_loader:
			z = model(x_test)
			_, yhat = torch.max(z.data, 1)

			correct += (yhat==y_test).sum().item()
		accuracy=correct/N_test
		accuracy_list.append(accuracy)
		print('Accuracy for the {}th epoch is {}'.format(epoch+1, accuracy), flush = True)

	with open('model/trained_model.pkl', 'wb') as f:
	    pickle.dump(model, f)	

else:
	if(not os.path.isfile('model/trained_model.pkl')):
		print('Train the model first')
		os._exit(1)

	with open('model/trained_model.pkl', 'rb') as f:
	    model = pickle.load(f)	


correct = 0
for x_test, y_test in validation_loader:
	z = model(x_test)
	_, yhat = torch.max(z.data, 1)

	correct += (yhat==y_test).sum().item()
accuracy=correct/N_test
accuracy_list.append(accuracy)
print('The final accuracy is {}'.format(accuracy), flush = True)