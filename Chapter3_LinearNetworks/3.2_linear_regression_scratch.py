import torch
import random

def synthetic_data(w, b, num_examples):
    # 生成数据
    X = torch.normal(0, 1, (num_examples, len(w))) # 生成num个x
    y = torch.matmul(X, w) + b # 加常数项&系数
    y += torch.normal(0, 0.01, y.shape) # 添加噪声
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 下标
    indices = list(range(num_examples))
    # 随机
    random.shuffle(indices)
    # start end stride
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # 将这个函数转为生成器（yield）
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    # lin模型
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    # MES
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    # 优化算法
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
    #     print(X, '\n', y)
    #     break

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
            l = loss(net(X=X, w=w, b=b), y=y)
            l.sum().backward()
            sgd([w,b], lr=lr, batch_size=batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w ,b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')