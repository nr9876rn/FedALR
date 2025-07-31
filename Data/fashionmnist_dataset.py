import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import DataLoader, Dataset
from Data.data_partition import partition_data

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "fashionmnist":
        train_dataset = FashionMNIST(base_path, train=True, download=True)
        test_dataset = FashionMNIST(base_path, train=False, download=True)
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))])

        
    train_image = train_dataset.data.numpy()
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data.numpy()
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
        val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Cifar_Truncated(data=train_image[val_idxs], labels=train_label[val_idxs], transform=transform_test)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions
