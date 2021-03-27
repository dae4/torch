from Xception import xception
import os
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision
import torch
import tqdm
from tqdm import trange

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
import copy
import time

# def StartTraining(dataRootDir, modelOutputDir, gpuNum, visibleGpu, batchSize, imgSize, epoch, release_mode=False):

ImageFile.LOAD_TRUNCATED_IMAGES = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'

dataRootDir=''
trainBaseDir = dataRootDir + "train"
testBaseDir = dataRootDir + "test"
modelOutputDir=""

def get_train_loader(image_size, batch_size, num_worker):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        # transforms.RandomAffine(45),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        ])
    train_datasets = torchvision.datasets.ImageFolder(root=trainBaseDir, transform=transform_train)
    test_data = torchvision.datasets.ImageFolder(root = testBaseDir,transform = transform_train) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    shuffle = False
    pin_memory = True
    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_worker, shuffle=shuffle, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(
        test_data,batch_size=batch_size,shuffle=False,num_workers=64)

    return train_loader ,testloader


def main_worker(gpu, ngpus_per_node):
    
    image_size = 299
    batch_size = 64*torch.cuda.device_count()
    num_worker = 128
    epochs = 1
    
   
    running_loss = 0.0
    running_corrects = 0
    best_acc = 0.0

    batch_size = int(batch_size / ngpus_per_node)
    num_worker = int(num_worker / ngpus_per_node)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)
    model = xception()
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = DDP(model, device_ids=[gpu])
 
    train_loader, test_loader = get_train_loader(
        image_size=image_size,
        batch_size=batch_size,
        num_worker=num_worker)
 
    dataset_sizes={}
    dataset_sizes['train']=len(train_loader)
    dataset_sizes['val']=len(test_loader)

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.001,
        momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(gpu)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.0005)
    
    phase='train'
    for epoch in range(epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                data_loader = train_loader
            else:
                model.eval()   # 모델을 평가 모드로 설정
                data_loader = test_loader

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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print(f'{epoch} epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        torch.save(model.state_dict(), modelOutputDir+f'{epoch}.pth')
    print('Best val Acc: {:4f}'.format(best_acc))

if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))
    print('Finished Training')
