
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)

#-------------------------------------------------Synthatic Data----------------------------------------------------------#
num_inputs = 2
num_examples = 1000
true_w = np.array([2, -3.4])
true_b = 4.2
features = np.random.normal(scale=1, size=(num_examples, num_inputs))
labels = np.dot(features, true_w) + true_b
labels += np.random.normal(scale=0.01, size=labels.shape)


# Create data splits
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=1234)
print ("X_train:", x_train.shape)
print ("y_train:", y_train.shape)
print ("X_test:", x_test.shape)
print ("y_test:", y_test.shape)

#-------------------------------------------------Model----------------------------------------------------------#
class RidgeRegression(nn.Module):
	def __init__(self, input_size, num_classess=1):
		super(RidgeRegression, self).__init__();
		self.linear = nn.Linear(input_size, num_classess)

	def forward(self, x):
		return self.linear(x)





#-------------------------------------------Model Configuration---------------------------------------------------#
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 2
num_epochs = 60
learning_rate = 0.001

# Initilize network
model = RidgeRegression(input_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# Convert numpy arrays to torch tensors
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# Train
for epoch in range(num_epochs):
	inputs = inputs.to(device=device)
	targets = targets.to(device=device)

	# data = data.reshape(data.shape[0], -1)

	# forward
	outputs = model(inputs)
	loss = criterion(outputs, targets)

	# backward
	optimizer.zero_grad()
	loss.backward()

	# next step
	optimizer.step()

	if (epoch+1) % 5 == 0:
		print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
print('Final Weights: {}'.format(model.linear.weight.data))
print('Final Bias: {}'.format(model.linear.bias.data))

