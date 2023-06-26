import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
    
#Nueral Network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Flatten(),
                                 nn.LazyLinear(256), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(128, 2), nn.Sigmoid())
        self.net.apply(init_cnn)
    
    def forward(self, X):
        return self.net(X)

#Xavier initialization
def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

#Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
num_epochs = 10
lr = 0.00009
batch_size = 32
side_length = 256
train_set_size = 10000
valid_set_size = 2000
test_set_size = 5000

#Loading in data: 0 = Cat; 1 = Dog
transform = transforms.Compose([transforms.Resize((side_length, side_length)),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(root='dataset/PetImages', transform=transform)

#Dataset was too large, so I remove some of the data using a partial dataser
partial_dataset, _, = data.random_split(dataset, [train_set_size + valid_set_size + test_set_size, 25000 - train_set_size - valid_set_size - test_set_size])
train_dataset, test_dataset = data.random_split(partial_dataset, [train_set_size + valid_set_size, test_set_size])
train_dataset, valid_dataset = data.random_split(train_dataset, [train_set_size, valid_set_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Init model, loss function, and optimizer
model = NN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

#Trainer
def trainer(num_epochs):
    for epoch in range(num_epochs):
        #Training
        model.train(True)
        train_loss = 0.0
        for data, labels in train_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            #Forward
            score = model(data)
            loss = criterion(score, labels)

            #Backpropogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #Gradient Desccent

            train_loss += loss.item()

        #Validating
        model.eval()
        valid_loss = 0.0
        for data, labels in valid_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            score = model(data)
            loss = criterion(score, labels)

            valid_loss += loss.item()

        print(f'Epoch {epoch + 1}:\tTraining Loss: {train_loss/len(train_loader)}\tValidation Loss: {valid_loss/len(valid_loader)}')

#Accuracy Checker
def check_accuracy(model, loader):
    if loader == train_loader:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

#Testing and Checking Accuracy
trainer(num_epochs=num_epochs)
check_accuracy(model=model, loader=train_loader)
check_accuracy(model=model, loader=test_loader)

torch.save(model.state_dict(), './dogcatmodel3_nn')

#Show an image
#plt.figure()
#f, axarr = plt.subplots(1,4)
#axarr[0].imshow(next(iter(train_loader))[0][0].permute(1, 2, 0))
#axarr[1].imshow(next(iter(train_loader))[0][0].permute(1, 2, 0))
#axarr[2].imshow(next(iter(train_loader))[0][0].permute(1, 2, 0))
#axarr[3].imshow(next(iter(train_loader))[0][0].permute(1, 2, 0))
#plt.show()