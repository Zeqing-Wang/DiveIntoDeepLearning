import math
import numpy as np
import torch
import sys
sys.path.append('..')
from tools.timer import Timer
# from d2l import torch as d2l
if __name__ == '__main__':
    timer = Timer()
    
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])


    # 矢量化加速
    c = torch.zeros(n)

    # 每次一位
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec') #0.05252 sec
    # 重载+
    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec') #0.00001 sec
    pass