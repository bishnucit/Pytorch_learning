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

print(torch.empty(2,3))
#tensor([[0., 0., 0.],[0., 0., 0.]])

print(torch.full((2, 3), 3.141592))
#tensor([[3.1416, 3.1416, 3.1416],[3.1416, 3.1416, 3.1416]])

x = torch.randn(2, 3)
print(x)
#tensor([[ 0.6580, -1.0969, -0.4614], [-0.1034, -0.5790,  0.1497]])
print(torch.cat((x, x, x), 0))
#tensor([[ 0.6580, -1.0969, -0.4614],
#        [-0.1034, -0.5790,  0.1497],
#        [ 0.6580, -1.0969, -0.4614],
#        [-0.1034, -0.5790,  0.1497],
#        [ 0.6580, -1.0969, -0.4614],
#        [-0.1034, -0.5790,  0.1497]])
print(torch.cat((x, x, x), 1))
#tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
#         -1.0969, -0.4614],
#        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
#         -0.5790,  0.1497]])

t = torch.tensor([[1,2],[3,4]])
print(torch.gather(t, 1, torch.tensor([[0,0],[1,0]])))
#tensor([[1, 1],
#        [4, 3]])

x = torch.randn(3, 4)
print(x)
#tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
#        [-0.4664,  0.2647, -0.1228, -1.1068],
#        [-1.1734, -0.6571,  0.7230, -0.6004]])
indices = torch.tensor([0, 2])
print(torch.index_select(x, 0, indices))
#tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
#        [-1.1734, -0.6571,  0.7230, -0.6004]])
print(torch.index_select(x, 1, indices))
#tensor([[ 0.1427, -0.5414],
#        [-0.4664, -0.1228],
#        [-1.1734,  0.7230]])

print(torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))) #returns indices of all non zero elements
#tensor([[0],[1],[2],[4]])

a = torch.arange(4.)
print(a)
#tensor([0., 1., 2., 3.])
print(torch.reshape(a,(2,2)))
#tensor([[0., 1.],
#        [2., 3.]])
b = torch.tensor([[0, 1], [2, 3]])
print(b)
#tensor([[0, 1],
#        [2, 3]])
print(torch.reshape(b, (-1,)))
#tensor([ 0,  1,  2,  3])

x = torch.randn(2,3)
print(x)
#tensor([[-0.2122,  1.1628, -0.7705],
#        [ 2.4844,  0.1818,  0.2871]])
print(torch.t(x)) #expects input to be matrix(2d tensor) and trasposes dimensions 0 an 1
#tensor([[-0.2122,  2.4844],
#        [ 1.1628,  0.1818],
#        [-0.7705,  0.2871]])

src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(torch.take(src, torch.tensor([0, 2, 5])))#Returns a new tensor with the elements of input at the given indices
#tensor([ 4,  5,  8])

x = torch.randn(2, 3)
print(x)
#tensor([[-0.7259,  1.8439,  0.1248],
#        [-0.2970, -1.2318,  0.1906]])
print(torch.transpose(x, 0, 1))
#tensor([[-0.7259, -0.2970],
#        [ 1.8439, -1.2318],
#        [ 0.1248,  0.1906]])

print(torch.unbind(torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]])))
#(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))

x = torch.randn(3, 2)
y = torch.ones(3, 2)
print(x)
#tensor([[ 0.6744, -0.7159],
#        [ 0.7594, -0.1935],
#        [ 1.2827, -0.3491]])
print(torch.where(x > 0, x, y))
#tensor([[0.6744, 1.0000],
#        [0.7594, 1.0000],
#        [1.2827, 1.0000]])

a= torch.empty(3,3).uniform_(0,1) #generate a uniform random matrix with range [0, 1]
print(a)
#tensor([[0.4388, 0.6387, 0.5247],
#        [0.6826, 0.3051, 0.4635],
#        [0.4550, 0.5725, 0.4980]])
print(torch.bernoulli(a))
#tensor([[0., 1., 1.],
#        [0., 0., 1.],
#        [1., 0., 1.]])

#Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
print(torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))) #
#tensor([ 1.8552,  1.2864,  3.8893,  3.4691,  5.0479,  5.9427,  6.3132,  8.6189,
#         9.3320, 10.1098])
print(torch.normal(mean=0.5, std=torch.arange(1., 6.)))
#tensor([ 0.2792,  0.1709, -4.3310, -3.5642,  6.1855])
print(torch.normal(mean=torch.arange(1., 6.)))
#tensor([1.9015, 0.4099, 2.0306, 4.6454, 6.1372])

#Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
torch.rand(4)
#tensor([0.2946, 0.8376, 0.9923, 0.9826])

torch.randn(2, 3)
#tensor([[-0.0700,  0.0358, -0.1230],
#        [ 0.6277, -0.3289, -1.6249]])

#Returns a random permutation of integers from 0 to n - 1.
torch.randperm(4)
#tensor([0, 1, 3, 2])

 # Save to file
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'tensor.pt')
# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.save(x, buffer)

torch.load('tensors.pt')
# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=torch.device('cpu'))
# Load all tensors onto the CPU, using a function
torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# Map tensors from GPU 1 to GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
# Load tensor from io.BytesIO object
with open('tensor.pt') as f:
    buffer = io.BytesIO(f.read())
torch.load(buffer)

#Math operations
torch.abs(torch.tensor([-1, -2, 3]))
#tensor([ 1,  2,  3])

a = torch.randn(4)
print(a)
#tensor([-0.4931,  0.1031, -1.3724,  0.7672])
torch.acos(a)
#tensor([2.0865, 1.4675,    nan, 0.6963])

a = torch.randn(4)
print(a)
#tensor([ 0.6939,  0.4375, -1.5540, -1.0776])
torch.add(a, 20)
#tensor([20.6939, 20.4375, 18.4460, 18.9224])

a = torch.randn(4)
b = torch.randn(4, 1)
torch.add(a, 10, b)
#tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
#        [-18.6971, -18.0736, -17.0994, -17.3216],
#        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
#        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])

a = torch.randn(4)
torch.ceil(a)
#tensor([-0., -1., -1.,  1.])





