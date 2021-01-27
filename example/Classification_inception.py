#%%
import torch
import numpy as np
import torchvision
import os

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,  models, transforms

use_cuda = torch.cuda.is_available()
BATCH_SIZE = 6
FINE_TUNE = False
# True 전체 네트워크 학습, False 최종 마지막 FC만

### Data Ready ###
train_transform = transforms.Compose([
    transforms.RandomSizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(300),
    transforms.CenterCrop(300),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder('image/train/',train_transform)
test_data = datasets.ImageFolder('image/train/',test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
#%%

## Data Visualize ##

def show():
    img, lab = next(iter(train_loader))
    class_name = [train_data.classes[i] for i in lab]
    img = torchvision.utils.make_grid(img,nrow=8)

    figsize(10,10)
    img = ing.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    image = std * img + mean
    if class_name is not None:
        plt.title(class_name)
    # plt.pause(0.001)

# show()
# not active...
#%%


inception = models.inception_v3(pretrained=True)

# Auxiliary 를 사용하지 않으면 inception v2와 동일
inception.aux_logits = False

# 일단 모든 layers를 requires_grad=False 를 통해서 학습이 안되도록 막습니다.
if not FINE_TUNE:
    for parameter in inception.parameters():
        parameter.requires_grad = False

# 새로운 fully-connected classifier layer 를 만들어줍니다. (requires_grad 는 True)
# in_features: 2048 -> in 으로 들어오는 feature의 갯수
n_features = inception.fc.in_features
inception.fc = nn.Linear(n_features, 2) 

if use_cuda:
    inception = inception.cuda()
    
criterion = nn.CrossEntropyLoss()

# Optimizer에는 requires_grad=True 인 parameters들만 들어갈수 있습니다.
optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, inception.parameters()), lr=0.001)


# %%
from tqdm.notebook import tqdm

def train_model(model, criterion,optimizer,epochs=30):
    for epoch in tqdm(range(epochs)):
        epoch_loss =0
        for step, (inputs, y_true) in enumerate(train_loader):
            if use_cuda:
                x_sample, y_true = inputs.cuda(), y_true.cuda()
            x_sample, y_true = Variable(x_sample), Variable(y_true)

            # optimizer 0으로 초기화
            optimizer.zero_grad()

            # Feedfoward
            y_pred = inception(x_sample)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            _loss = loss.data[0]
            epoch_loss +=loss

        print(f'[{epoch+1}] loss:{epoch_loss/step:.4f}')

train_model(inception, criterion,optimizer)
# %%
def validate(model, epochs=1):
    model.train(False)
    n_total_correct = 0
    for step, (inputs, y_true) in enumerate(test_loader):
        if use_cuda:
            x_sample, y_true = inputs.cuda(), y_true.cuda()
        x_sample, y_true = Variable(x_sample), Variable(y_true)

        y_pred = model(x_sample)
        _, y_pred = torch.max(y_pred.data, 1)
        
        n_correct = torch.sum(y_pred == y_true.data)
        n_total_correct += n_correct
        
    print('accuracy:', n_total_correct/len(test_loader.dataset))

validate(inception)