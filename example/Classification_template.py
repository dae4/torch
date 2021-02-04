#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from tqdm.notebook import tqdm

## Check device
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#%%
## data_set : ( https://pytorch.org/docs/stable/torchvision/datasets.html)

data_dir = "FashionMNIST/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = trainset.classes
print(trainset.class_to_idx)
#%%
## check the data.shape
## (batch,w,h,c)
print(trainloader.dataset.data.shape)
print(testloader.dataset.data.shape)
#%%

## model ( https://pytorch.org/docs/stable/torchvision/models.html )

model_dir = 'model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

import torchvision.models as model
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft = model_ft.to(device)

# #%%
# show model.summary like keras
# from torchsummary import summary
# print(summary(model_ft,(3,32,32)),device="cuda:2")

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#%%

for epoch in tqdm(range(2)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # get GPU input 
        inputs,labels = inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#%%

## only parameter save 
torch.save(model.state_dict(),model_dir)

## model save
#torch.save(model,model_dir)
#%%
# Model Load 
## 
from torchvision.models.resnet import Resnet

save_model = model_dir +'.pth'

loaded_model = ResNet()

loaded_model.load_state_dict(torch.load(save_model))

## saved model with parameter

# the_model = torch.load(save_model)