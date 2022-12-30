import torch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x+x, x-y, x*y, x/y, x**y)
print(torch.exp(x))
print(x.sum())

# 广播机制
# a复制列 b复制行 两者再相加
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a+b)