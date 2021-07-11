import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#-------------------------------------------------Model----------------------------------------------------------#
class LeNet5(nn.Module):
	def __init__(self, in_channels = 1, num_classes = 10):
		super(LeNet5, self).__init__()




#-------------------------------------------Model Configuration---------------------------------------------------#
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classses = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initilize network
model = LeNet5(input_size, num_classses).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
	model.train()
	for batch_index, (data, targets) in enumerate(train_loader):
		data = data.to(device=device)
		targets = targets.to(device=device)

		data = data.reshape(data.shape[0], -1)

		# forward
		scores = model(data)
		loss = criterion(scores, targets)

		# backward
		optimizer.zero_grad()
		loss.backward()

		# next step
		optimizer.step()

# Test
def check_accuracy(loader, model):
	if loader.dataset.train:
		print('Checking accuracy on traning data')
	else:
		print('Checking accuracy on testing data')

	num_correct = 0
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)
			x = x.reshape(x.shape[0], -1)

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

			print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
	model.train()




check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

