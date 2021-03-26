import torch.nn as nn
from Xception import xception
import torch.optim as optim
import os
import shutil
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm import trange
from torch.optim import lr_scheduler
import copy
import time

ImageFile.LOAD_TRUNCATED_IMAGES = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 299
batch_size = 32 * torch.cuda.device_count()
epochs=50


dataRootDir=''
trainBaseDir = dataRootDir + "train"
testBaseDir = dataRootDir + "test"

classes_name = sorted(os.listdir(trainBaseDir))
modelOutputDir= ""

shutil.rmtree(modelOutputDir, ignore_errors=True)
if not os.path.exists(modelOutputDir):
    os.makedirs(modelOutputDir)

trans = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root = trainBaseDir,transform = trans)
test_data = torchvision.datasets.ImageFolder(root = testBaseDir,transform = trans)    

trainloader= DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=32)
testloader = DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=32)


## Create Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xception()
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(),"GPUs!")
    model = torch.nn.DataParallel(model)

net=model.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()



# 5 에폭마다 0.1씩 학습률 감소
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.0005)

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

dataset_sizes = len(train_data)+len(test_data)


for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 모델을 학습 모드로 설정
            data_loader = trainloader
        else:
            model.eval()   # 모델을 평가 모드로 설정
            data_loader = testloader
        running_loss = 0.0
        running_corrects = 0

        # 데이터를 반복
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 매개변수 경사도를 0으로 설정
            optimizer.zero_grad()

            # 순전파
            # 학습 시에만 연산 기록을 추적
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 학습 단계인 경우 역전파 + 최적화
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # 모델을 깊은 복사(deep copy)함
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# 가장 나은 모델 가중치를 불러옴
model.load_state_dict(best_model_wts)

print('Finished Training')