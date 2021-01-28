import torch
import numpy as np

# Tensor Initialization 

# directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# with random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# shape, dataype, and device
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    print('GPU available!')
    tensor = tensor.to('cuda')
    
# standard numpy like indexing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# multiplying tensors
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# in-place operation
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# tensor to numpy array - if tnesor is in CPU
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# a change in the tensor reflects in the numpy
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numoy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# changes in the numpy array reflects in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")