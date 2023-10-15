# -*- coding: UTF-8 -*-
# @author : jianfei.zhao
# @date : 2023-10-15
# @description : CIFAR 10 dataset  test on dog picture
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src import CIFAR10M

test_data = CIFAR10(root="data", train=False, transform=transforms.ToTensor(), download=True)
query_dict = test_data.class_to_idx


def get_key(val):
    for key, value in query_dict.items():
        if val == value:
            return key


img_path = "data/dog.jpg"
image = Image.open(img_path)
transforms1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
image = transforms1(image)
image = torch.reshape(image, (1, 3, 32, 32))

model1 = CIFAR10M()
states_dict = torch.load("pth/trained_CIFAR10_state_20.pth")
model1.load_state_dict(states_dict)
model1.eval()
with torch.no_grad():
    output = model1(image)
    index = output.argmax(1).item()
    print(get_key(index))
