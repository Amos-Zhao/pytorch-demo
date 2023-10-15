# -*- coding: UTF-8 -*-
# @author : jianfei.zhao
# @date : 2023-10-13
# @description : CIFAR 10 dataset train the model
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from datetime import datetime
from src.model import CIFAR10M

start = datetime.now()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = CIFAR10(root="data", train=True, transform=transforms.ToTensor(), download=True)
test_data = CIFAR10(root="data", train=False, transform=transforms.ToTensor(), download=True)
print(f"size of train data: {len(train_data)}")
print(f"size of test  data: {len(test_data)}")
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# instantiation a model
test = CIFAR10M().to(device)
loss = CrossEntropyLoss().to(device)
# optimizer
learning_rate = 0.01
optimizer = SGD(test.parameters(), lr=learning_rate)
# model parameters
train_count = 0
test_count = 0
epoch = 20
# tensorboard --logdir=log_seq
writer = SummaryWriter("log_seq")
for i in range(epoch):
    print(f"the number of training rounds is: {i + 1}")
    for data in train_data_loader:
        images, targets = data
        images, targets = images.to(device), targets.to(device)
        outputs = test(images)
        train_loss = loss(outputs, targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_count += 1
        # running_loss += result_loss
        writer.add_scalar(tag='train', scalar_value=train_loss, global_step=train_count)
        if train_count % 100 == 0:
            print(f"train count is:{train_count}, loss is:{train_loss.item()}")
    with torch.no_grad():
        total_test_loss = 0.0
        total_accuracy = 0
        for data in test_data_loader:
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            outputs = test(images)
            test_loss = loss(outputs, targets)
            test_count += 1
            total_test_loss += test_loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
        writer.add_scalar(tag='test_loss', scalar_value=total_test_loss, global_step=i)
        writer.add_scalar(tag='test_accuracy', scalar_value=total_accuracy / len(test_data), global_step=i)
        print(f"epoch is:{i + 1} loss in test data set:{total_test_loss}")
        print(f"epoch is:{i + 1} accuracy in test data set:{total_accuracy / len(test_data):.2%}")
    # recommended way to store model details
    torch.save(test.state_dict(), f"pth/trained_CIFAR10_state_{i + 1}.pth")
writer.close()
end = datetime.now()
print(f"total running time is:{end - start}")
