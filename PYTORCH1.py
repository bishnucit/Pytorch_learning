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
