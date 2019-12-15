from helpers import *

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('--path', required = True)

args = vars(parser.parse_args())

if(not os.path.exists(args['path'])):
	print('Not a valid path')
	os._exit(1)

if(not os.path.isfile(args['path'])):
	print('Not a file')
	os._exit(1)

if(not os.path.isfile('model/trained_model.pkl')):
	print('Train the model first')
	os._exit(1)

with open('model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

image = Image.open(args['path']).convert('LA')
image = np.array(image, dtype= 'uint8')[:,:,0]
if abs(image[0,0] - 255) < 20:
	image = 255-image
image = Image.fromarray(image, 'L')
image = image.resize((16, 16))
trans = transforms.ToTensor()
softmax_fn=nn.Softmax(dim=-1)
x = trans(image)
x = x.unsqueeze(0)
z = model(x)
z = softmax_fn(z)
probability, yhat = torch.max(z, 1)

print("Prediction: {}".format(int(yhat[0])))
print("Probability: {}".format(float(probability[0])))