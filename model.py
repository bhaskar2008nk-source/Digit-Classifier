import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Small CNN for 28x28 grayscale digits."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.f1 = nn.Linear(64 * 7 * 7, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = self.drop(torch.flatten(x, 1))
        return self.f2(F.relu(self.f1(x)))
