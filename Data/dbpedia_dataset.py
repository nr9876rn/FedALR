import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import DBpedia
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from Data.data_partition import partition_data
import os
import json

class SentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class, flag, task_num, task_idx,
                 dataidx_map=None):
    # 确保中文分词效果更好
    tokenizer = get_tokenizer('basic_english')

    # --------------------------
    # 核心修改：加载word_map.json，与模型保持一致
    # --------------------------
    word_map_path = os.path.join("../Data/Dataset/dbpedia/sents", 'word_map.json')
    if not os.path.exists(word_map_path):
        raise FileNotFoundError(f"未找到word_map.json，请先运行generate_word_map生成")

    with open(word_map_path, 'r', encoding='utf-8') as j:
        word_map = json.load(j)

    # 从word_map中获取特殊标记的索引（必须与模型一致）
    pad_idx = word_map['<pad>']
    unk_idx = word_map['<unk>']
    vocab_size = len(word_map)
    print(f"从word_map加载词汇表，大小：{vocab_size}（与模型一致）")

    with open(base_path + '/dbpedia/processed_data/processed_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    train_data = dataset['train_data']
    train_label = np.array(dataset['train_label'])  # 转回numpy数组
    test_data = dataset['test_data']
    test_label = np.array(dataset['test_label'])
    print(f"已从 {base_path + '/dbpedia/processed_data/processed_dataset.json'} 加载数据集")

    # 转换为numpy数组
    train_data = np.array(train_data, dtype=object)
    train_label = np.array(train_label)
    test_data = np.array(test_data, dtype=object)
    test_label = np.array(test_label)

    n_train = train_label.shape[0]

    if flag == 0:
        net_dataidx_map = partition_data(partition, n_train, n_parties * task_num, train_label, beta, skew_class,
                                         task_num, task_idx)
    else:
        net_dataidx_map = dataidx_map

    temp_map = {}
    # id_idx = [[0, 1], [0, 2], [0, 3], [0, 4]]  # 保留
    # id_idx = [[0, 1], [0, 2], [1, 3], [2, 4]]  # 慢丢弃
    id_idx = [[0, 1], [1, 2], [2, 3], [3, 4]]  # 快丢弃
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

    # 创建数据加载器（注意padding_value要与word_map的<pad>索引一致）
    train_dataloaders = []
    val_dataloaders = []

    for i in range(n_parties):
        train_idxs = temp_map[i][:int(1.0 * len(temp_map[i]))]
        val_idxs = temp_map[i][int(1.0 * len(temp_map[i])):]

        if len(train_idxs) == 0:
            train_idxs = val_idxs[:1]
            val_idxs = val_idxs[1:]

        train_dataset = SentDataset(data=train_data[train_idxs], labels=train_label[train_idxs])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: text_collate_fn(batch, pad_idx)  # 传入pad_idx
        )

        val_dataset = SentDataset(data=train_data[val_idxs], labels=train_label[val_idxs])
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: text_collate_fn(batch, pad_idx)
        )

        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)

    test_dataset = SentDataset(data=test_data, labels=test_label)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: text_collate_fn(batch, pad_idx)
    )

    return train_dataloaders, val_dataloaders, test_loader, temp_map, traindata_cls_counts, net_dataidx_map


# 修改collate_fn以支持动态pad_idx
def text_collate_fn(batch, pad_idx):
    texts, labels = zip(*batch)
    text_tensors = [torch.tensor(text) for text in texts]
    padded_texts = pad_sequence(text_tensors, batch_first=True, padding_value=pad_idx)  # 使用word_map中的pad索引
    labels = torch.tensor(labels)
    return padded_texts, labels

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        if not dataidx:  # 处理空数据索引
            net_cls_counts_dict[net_i] = {c: 0 for c in range(num_classes)}
            continue

        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp

        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate((net_cls_counts_npy, tmp_npy), axis=0)

    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))

    data_list = []
    for net_id, data in net_cls_counts_dict.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)

    print('数据统计信息:')
    print(f'平均每个客户端样本数: {np.mean(data_list):.2f}')
    print(f'样本数标准差: {np.std(data_list):.2f}')
    print(f'数据分布: {str(net_cls_counts_dict)}')
    print(f'数据分布矩阵:\n{net_cls_counts_npy.astype(int)}')

    return net_cls_counts_npy