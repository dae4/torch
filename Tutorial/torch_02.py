#%%
import torch

# %%
# Add Operations
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
print(x) 
y = torch.rand(5,3)
print(x+y)

# Add operations 2
result = torch.empty(5, 3)
torch.add(x,y,out=result)
print(result)
# %%
# Add operataion 3
# 바꿔치기(in-place) 방식으로 tensor의 값을 변경하는 연산 뒤에는 _``가 붙습니다. 
# 예: ``x.copy_(y), x.t_() 는 x 를 변경합니다.
y.add_(x)
print(y)

# %%
# axis = 1 
print(x[:, 1])
# %%
x = torch.randn(4,4)
y = x.view(16) # 16
z = x.view(-1,8) # x ,8
print(x.size(),y.size(),z.size())
# %%

x= torch.rand(1)
print(x)
print(x.item()) ## 값만 불러오기 

# %%
# Torch tensor -> numpy 

a=torch.ones(5)
print(a)

b=a.numpy()
print(b)

a.add_(1)
print(a)
print(b)
# %%
# CUDA Tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)
    y = torch.ones_like(x, device=device)  # directly create tensor in gpu 
    x = x.to(device)                       
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))      
# %%
