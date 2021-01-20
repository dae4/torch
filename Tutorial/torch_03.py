#%%
# AUTOGRAD 
import torch

x = torch.ones(2,2,requires_grad=True)
# requires_grad 
# 그 tensor에서 이뤄진 모든 연산들을 추적함
print(x)
# %%
y = x+2
print(y)
# %%
# y는 연산 결과로 생성되어 grad_fn를 갖음
# grad_fn는 tensor를 생성한 funcion을 참조함
print(y.grad_fn)

# %%
# (1+2)*(1+2)*3=27
z= y*y*3
out = z.mean()
print(z,out)
# %%
# .requires_grad_는 .requires_grad 값을 바꿔치기(in-place)함

a = torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)
# %%
## Gradient
## *** realate backprop
## out = z.mean()
out.backward()
## out.backward(torch.tensor(1))
print(x.grad) ## dout/dx
# %%

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000: 
    ## y.data.norm ==> torch.sqrt(torch.sum(torch.pow(y, 2)))
    y = y*2
print(y)

# %%
gradients  = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients )

print(x.grad)
# %%
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
# %%
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
# %%
