# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        # Conv2d(input,output,kernel_size(x,x))
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)

        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
# %%

params = list(net.parameters())
print(len(params))
print(params[0].size())
# %%
## input 임의 생성
input = torch.rand(1,1,32,32)
print(input)
out = net(input)
print(out)
# %%

## zero_grad() 로 이전 gradient history를 갱신함
net.zero_grad() ## 역전파 전 grad를 0으로 갱신
out.backward(torch.randn(1, 10))
# %%
# Loss Function
output = net(input)
target = torch.randn(10)
target = target.view(1,-1) # 출력과 같은 shape
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)
# %%
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# %%

## Backprop
net.zero_grad()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

## apply backward
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# %%
