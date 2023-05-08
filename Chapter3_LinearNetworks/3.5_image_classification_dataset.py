import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import sys
sys.path.append('..')
from tools.timer import Timer

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

if __name__ == '__main__':
    batch_size = 256
    num_workers = 4
    resize = 64

    timer = Timer()
    trans = [transforms.ToTensor(), transforms.Resize(resize)]
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    print(len(mnist_train), len(mnist_test))
    
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'{timer.stop():.5f} sec') #0.00001 sec
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
    pass