import torch.nn as nn
import torch
import os
import json


# class SimpleCNN(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim=10):
#         super(SimpleCNN, self).__init__()
#         self.base1 = Base1()  # 卷积层
#         self.base2 = Base2(input_dim, hidden_dims)  # 全连接层
#         self.classifier = Classifier(hidden_dims[1], output_dim)
#
#     def forward(self, x):
#         x = self.base1(x)
#         x = self.base2(x)
#         return self.classifier(x)
#
# class Base1(nn.Module):
#     def __init__(self):
#         super(Base1, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         return x
#
# class Base2(nn.Module):
#     def __init__(self, input_dim, hidden_dims):
#         super(Base2, self).__init__()
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(input_dim, hidden_dims[0])
#         self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
#
#     def forward(self, x):
#         x = x.view(-1, 16 * 5 * 5)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         return x
#
# class Classifier(nn.Module):
#     def __init__(self, hidden_dims, output_dim=10):
#         super(Classifier, self).__init__()
#         self.fc3 = nn.Linear(hidden_dims, output_dim)
#
#     def forward(self, x):
#         x = self.fc3(x)
#         return x
#
# def cifar_cnn(n_classes):
#     return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)



class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))


class FE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        x = self.fc3(x)
        return x

def cifar_cnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)





# class AlexNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 4 * 4, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
#
# def alexnet(n_classes):
#     return AlexNet(num_classes=n_classes)

