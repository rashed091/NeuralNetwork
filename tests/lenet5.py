# <ul>
#   <li><strong>Input</strong>: Gray scale image of size 32 x 32.</li>
#   <li><strong>C1</strong>: Convolutional layer of 6 feature maps, kernel size (5, 5) and stride 1. Output size therefore is 6 X 28 x 28. Number of trainable parameters is $(5*5 + 1) * 6 = 156$.</li>
#   <li><strong>S2</strong>: Pooling/subsampling layer with kernel size (2, 2) and stride 2. Output size is 6 x 14 x 14. Number of trainable parameters = 0.</li>
#   <li><strong>C3</strong>: Convolutional layer of 16 feature maps. Each feature map is connected to all the 6 feature maps from the previous layer. Kernel size and stride are same as before. Output size is 16 x 10 x 10. Number of trainable parameters is $(6 * 5 * 5 + 1) * 16 = 2416$.</li>
#   <li><strong>S4</strong>: Pooling layer with same <em>hyperparameters</em> as above. Output size = 16 x 5 x 5.</li>
#   <li><strong>C5</strong>: Convolutional layer of 120 feature maps and kernel size (5, 5). This amounts to <em>full connection</em> with outputs of previous layer. Number of parameters are $(16 * 5 * 5 + 1)*120 = 48120$.</li>
#   <li><strong>F6</strong>: <em>Fully connected layer</em> of 84 units. i.e, All units in this layer are connected to previous layer’s outputs<span id="fc" class="margin-toggle sidenote-number"></span><span class="sidenote">This is same as layers in MLP we’ve seen before.</span>. Number of parameters is $(120 + 1)*84 = 10164$</li>
#   <li><strong>Output</strong>: Fully connected layer of 10 units with softmax activation<span id="out" class="margin-toggle sidenote-number"></span><span class="sidenote">Ignore ‘Gaussian connections’. It is for a older loss function no longer in use.</span>.</li>
# </ul>

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as data
import matplotlib.pyplot as plt
import numpy as np

train_dataset = data.MNIST(root='./data', train=True, transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]), 
                                           download=True)

test_dataset = data.MNIST(root='./data', train=False, transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 120, kernel_size=5),
                nn.ReLU())

        self.fc = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10), nn.LogSoftmax(dim=1))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output


model = LeNet5()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images
        images = images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images
                images = images.requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(
                iter, loss.item(), accuracy))
