import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Data.data_partition import partition_data
import torch
import os
from torchtext.datasets import SogouNews
from torchtext.data.functional import to_map_style_dataset

class SentDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    

def dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class, flag, task_num, task_idx, dataidx_map):
    if dataset == "yahoo":
        traindata = torch.load(os.path.join('../Data/Dataset/yahoo_answers_csv/sents/TRAIN_data.pth.tar'))
        testdata = torch.load(os.path.join('../Data/Dataset/yahoo_answers_csv/sents/TEST_data.pth.tar'))
        train_data = np.array(traindata['sents'])
        train_label = np.array(traindata['labels'])
        test_data = np.array(testdata['sents'])
        test_label = np.array(testdata['labels'])

        n_train = train_label.shape[0]
        if flag == 0:
            net_dataidx_map = partition_data(partition, n_train, n_parties * task_num, train_label, beta, skew_class,
                                             task_num, task_idx)
        else:
            net_dataidx_map = dataidx_map

        temp_map = {}
        # id_idx = [[0, 1], [0, 2], [0, 3], [0, 4]]  # 保留
        id_idx = [[0, 1], [0, 2], [1, 3], [2, 4]]  # 慢丢弃
        # id_idx = [[0, 1], [1, 2], [2, 3], [3, 4]]  # 快丢弃
        for i in range(n_parties):
            temp_map[i] = []
            # for id in range(task_idx+1):
            for id in range(id_idx[task_idx][0], id_idx[task_idx][1]):
                temp_map[i] += net_dataidx_map[id + i * task_num]
        traindata_cls_counts = record_net_data_stats(train_label, temp_map)

        temp2_map = {}
        for i in range(n_parties):
            temp2_map[i] = []
            for id in range(task_idx + 1):
                temp2_map[i] += net_dataidx_map[id + i * task_num]
        traindata_cls_counts = record_net_data_stats(train_label, temp2_map)

        train_dataloaders = []
        val_dataloaders = []
        for i in range(n_parties):
            train_idxs = temp_map[i][:int(1 * len(temp_map[i]))]
            val_idxs = temp_map[i][int(1 * len(temp_map[i])):]
            train_dataset = SentDataset(data=train_data[train_idxs], labels=train_label[train_idxs])
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = SentDataset(data=train_data[val_idxs], labels=train_label[val_idxs])
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            train_dataloaders.append(train_loader)
            val_dataloaders.append(val_loader)

        test_dataset = SentDataset(data=test_data, labels=test_label)
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
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))

    data_list = []
    for net_id, data in net_cls_counts_dict.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy.astype(int))

    return net_cls_counts_npy