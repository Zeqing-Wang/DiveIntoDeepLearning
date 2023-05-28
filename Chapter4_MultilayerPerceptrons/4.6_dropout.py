import torch
from torch import nn
def dropout_layer(X, dropout):
    # 检测是否合理
    assert 0 <= dropout <= 1
    # 全部drop
    if dropout == 1:
        return torch.zeros_like(X)
    # 全部不drop
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


# concise
dropout1 = 0.5
dropout2 = 0.6

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Dropout(256, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10)
                    )