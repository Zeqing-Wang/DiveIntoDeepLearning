import torch
A = torch.arange(20).reshape(5, 4)
print(A)
# 矩阵转置
print(A.T)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B)

# 求sum  0是纵轴 1是横轴
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

# 范数！
# 可以理解为向量分量的大小
u = torch.tensor([3.0, -4.0])
# L2
print(torch.norm(u))
# L1
print(torch.abs(u).sum())