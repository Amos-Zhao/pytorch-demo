# -*- coding: UTF-8 -*-
# @author : jianfei.zhao
# @date : 2023-10-15
# @description : a model for CIFAR10
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, Module


class CIFAR10M(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
