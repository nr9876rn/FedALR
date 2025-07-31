import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import DataLoader, Dataset
from Data.data_partition import partition_data

class SVHN_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(SVHN_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # SVHN数据需要从CHW转换为HWC格式
        img = np.transpose(img, (1, 2, 0))
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class, flag, task_num, task_idx, dataidx_map):
    if dataset == "svhn":
        # SVHN数据集使用download=True参数时会自动下载
        train_dataset = SVHN(base_path+'/svhn', split='train', download=True)
        test_dataset = SVHN(base_path+'/svhn', split='test', download=True)

        # SVHN数据集的预处理
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

     # 处理SVHN数据集格式
    train_image = train_dataset.data
    train_label = np.array(train_dataset.labels)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.labels)
    n_train = train_label.shape[0]

    if flag == 0:
        net_dataidx_map = partition_data(partition, n_train, n_parties*task_num, train_label, beta, skew_class, task_num, task_idx)
    else:
        net_dataidx_map = dataidx_map

    # sorted_dataidx_map = {key: sorted(net_dataidx_map[key]) for key in net_dataidx_map}
    # temp_map = {key: sorted_dataidx_map[key][0:int((task_idx + 1) * (len(sorted_dataidx_map[key]) / task_num))] for key in
    #             sorted_dataidx_map}
    temp_map = {}
    # id_idx = [[0, 1], [0, 2], [0, 3], [0, 4]]  # 保留
    id_idx = [[0, 1], [0, 2], [1, 3], [2, 4]]  # 慢丢弃
    # id_idx = [[0, 1], [1, 2], [2, 3], [3, 4]]  # 快丢弃
    for i in range(n_parties):
        temp_map[i] = []
        # for id in range(task_idx+1):
        for id in range(id_idx[task_idx][0], id_idx[task_idx][1]):
            temp_map[i] += net_dataidx_map[id + i*task_num]
    traindata_cls_counts = record_net_data_stats(train_label, temp_map)

    temp2_map = {}
    for i in range(n_parties):

        temp2_map[i] = []
        for id in range(task_idx+1):
            temp2_map[i] += net_dataidx_map[id + i * task_num]
    traindata_cls_counts = record_net_data_stats(train_label, temp2_map)

    train_dataloaders = []
    val_dataloaders = []
    for i in range(n_parties):
        train_idxs = temp_map[i][:int(1*len(temp_map[i]))]
        val_idxs = temp_map[i][int(1*len(temp_map[i])):]
        train_dataset = SVHN_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = SVHN_Truncated(data=train_image[val_idxs], labels=train_label[val_idxs], transform=transform_test)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    test_dataset = SVHN_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloaders, val_dataloaders, test_loader, temp_map, traindata_cls_counts, net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))


    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy.astype(int))

    return net_cls_counts_npy

