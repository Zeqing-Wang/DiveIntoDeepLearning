# only concise

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

def train_concise(wd):
    num_inputs = 10
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([
        {"params":net[0].weight, 'weight_decay':wd},
        {"params":net[0].bias}], lr = lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backwar()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            print(epoch+1, loss.item())
    pass

if __name__ == '__main__':
    
    pass