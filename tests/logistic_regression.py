import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as data
import matplotlib.pyplot as plt
import numpy as np

train_dataset = data.MNIST(root='./data', train=True,
                           transform=transforms.ToTensor(), download=True)

test_dataset = data.MNIST(root='./data', train=False,
                          transform=transforms.ToTensor())

batch_size = 100
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# img = train_dataset[0][0].numpy().reshape(28, 28)
# print(train_dataset[0][1])
# plt.imshow(img)
# plt.show()


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


epochs = int(3000 / (len(train_dataset) / batch_size))
input_dim = 28 * 28
output_dim = 10
learning_rate = 0.001

model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


iter = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).requires_grad_()
        labels = labels

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28 * 28). requires_grad_()

                outputs = model(images)

                _, predict = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predict == labels).sum()

            accuracy = 100 * (correct.item() / total)

            print('Iteration: {}, Loss: {}, Accuracy: {}'.format(
                iter, loss.item(), accuracy))

save_model = True
if save_model:
    torch.save(model, 'logistic_model.pth')
