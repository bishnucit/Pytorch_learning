"""
Autograd - Automatic differentiation

Autograd package Provide automatic differentiation for all operations on Tensors.
It is a define by run framework, the backdrop is defined by how the code runs, every
single iteration can be different.

Tensor - 

torch.Tensor is the central class of the package.
if its attribute is set as .requires_grad as True, it starts to track all operations on it.
After finishing computation when .backward() is called, it will automatically compute the gradients which will
be accumulated in .grad attribute. To stop tracking history, .detach() can be called

To prevent tracking history, save memory, torch.no_grad(): can be used to wrap the code.

Another important class for Autograd is Function.

Function -

Tensor and Function are interconnected and build up an acyclic graph, 
that encodes a complete history of computation. Each tensor has a .grad_fn 
attribute that references a Function that has created the Tensor (except for 
Tensors created by the user - their grad_fn is None).

import torch
Create a tensor and set requires_grad=True to track computation with it

x = torch.ones(2, 2, requires_grad=True)
print(x)
Out:

tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
Do an operation of tensor:

y = x + 2
print(y)
Out:

tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
y was created as a result of an operation, so it has a grad_fn.

print(y.grad_fn)
Out:

<AddBackward0 object at 0x7f0ea616bac8>
Do more operations on y

z = y * y * 3
out = z.mean()

print(z, out)
Out:

tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward1>)
.requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. The input flag defaults to False if not given.

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
Out:

False
True
<SumBackward0 object at 0x7f0e86396e48>
GRADIENTS
Let’s backprop now Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1)).

out.backward()
print gradients d(out)/dx

print(x.grad)
Out:

tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
You should have got a matrix of 4.5. Let’s call the out Tensor “o”. We have that o=14∑izi, zi=3(xi+2)2 and zi∣∣xi=1=27. Therefore, ∂o∂xi=32(xi+2), hence ∂o∂xi∣∣xi=1=92=4.5.

You can do many crazy things with autograd!

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
Out:

tensor([-1178.9551,  1202.9015,   293.6342], grad_fn=<MulBackward0>)
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)
Out:

tensor([ 102.4000, 1024.0000,    0.1024])
You can also stop autograd from tracking history on Tensors with .requires_grad=True by wrapping the code block in with torch.no_grad():

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
Out:

True
True
False
"""
