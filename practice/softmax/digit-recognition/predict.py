import pickle
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms
import argparse

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
image = image.resize((28, 28))
trans = transforms.ToTensor()
x = trans(image)
z = model(x.reshape(-1, 28*28))
_, yhat = torch.max(z, 1)

print("Prediction: {}".format(int(yhat[0])))