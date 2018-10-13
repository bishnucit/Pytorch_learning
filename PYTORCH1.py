 #Tensors - similar to numpy ndarrays also can be used on a GPU accelerated computing
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
x = torch.empty(5, 3) #5x3 uninitialized matrix 
print(x)

x = torch.rand(5,3) # 5x3 randomly initialized matrix:
print(x)

x = torch.zeros(5,3, dtype=torch.long) #matrix filled zeros and of dtype long
print(x)

x = torch.tensor([5.5,3]) #tensor directly from data
print(x)


x = x.new_ones(5, 3, dtype=torch.double) #tensor based on an existing tensor. 
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

#get x size
print(x.size())

#operatations
y = torch.rand(5,3)
print (x+y)
print(torch.add(x, y))

# providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


# adds x to y vAny operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print(y)

#If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#f you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

#Converting a Torch Tensor to a NumPy Array

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

#Converting NumPy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


print(torch.is_tensor(torch.tensor([1,2])))
#True

print(torch.is_storage(torch.tensor([1,2])))
#False

torch.tensor([1.2, 3]).dtype
#torch.float32
torch.set_default_dtype(torch.float64)
print(torch.tensor([1.2, 3]).dtype)
#torch.float64

print(torch.get_default_dtype())
#torch.floaat32


print(torch.numel(torch.randn(1,2,3,4,5)))
#120
a=torch.zeros(4,4)
print(torch.numel(a))
#16

print(torch.zeros_like(torch.empty(2,3)))
#tensor([[0., 0., 0.],
#        [0., 0., 0.]])

print(torch.ones(2,3))
#tensor([[1., 1., 1.],[1. ,1., 1.]])

print(torch.ones(3))
#tensor([1., 1., 1.])

input=torch.empty(2,3)
print(torch.ones_like(input)) 
#tensor([[1., 1., 1.],[1., 1., 1.]])

print(torch.arange(5))
#tensor([0, 1, 2, 3, 4])
print(torch.arange(1,4))
#tensor([1, 2, 3])
print(torch.arange(1,2.5,0.5))
#tensor([1.0000, 1.5000, 2.0000])

 
#Returns a one-dimensional tensor of steps equally spaced points between start and end.
print(torch.linspace(3, 10, steps=5))
#tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000]) 
print(torch.linspace(-10, 10, steps=5))
#tensor([-10.,  -5.,   0.,   5.,  10.])
print(torch.linspace(start=-10, end=10, steps=5))
#tensor([-10.,  -5.,   0.,   5.,  10.])

print(torch.eye(3))
#tensor([[1.,0.,0.],[0., 1., 0.], [0. ,0., 1.]])
