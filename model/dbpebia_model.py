import json
import torch
import torch.nn as nn
import os
from torchtext.vocab import build_vocab_from_iterator, FastText, GloVe
from torchtext.data.utils import get_tokenizer


class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=14):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        return self.fc3(x)


class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)  # 关键：指定padding_idx为<pad>的索引（0）
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=100,
                kernel_size=(size, emb_size)
            )
            for size in [3, 4, 5]
        ])
        self.relu = nn.ReLU()

    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)  # (batch_size, 1, word_pad_len, emb_size)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = torch.cat(pooled, dim=1)  # (batch size, n_kernels * len(kernel_sizes))
        return flattened


class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, emb_size)
        self.classifier = Classifier(300, n_classes)

    def forward(self, x):
        return self.classifier(self.base(x))


def dbpedia_textcnn(n_classes=14, emb_size=50):  # 关键：接收vocab_size参数
    """通过word_map.json构建词汇表并初始化模型"""
    # 1. 加载word_map.json
    word_map_path = os.path.join("../Data/Dataset/dbpedia/sents", 'word_map.json')  # 注意路径与生成时一致
    if not os.path.exists(word_map_path):
        raise FileNotFoundError(f"未找到word_map.json，请先运行generate_word_map生成")

    with open(word_map_path, 'r', encoding='utf-8') as j:
        word_map = json.load(j)

    # 2. 构建与word_map一致的vocab（确保索引对应）
    # 从word_map中提取词汇（按索引排序，确保顺序正确）
    sorted_words = sorted(word_map.keys(), key=lambda x: word_map[x])
    vocab = build_vocab_from_iterator([sorted_words], specials=[])  # 不重复添加特殊标记（已在word_map中）
    # 未登录词映射到<unk>（索引1）
    vocab.set_default_index(word_map['<unk>'])

    # 3. 词汇表大小（与word_map长度一致）
    vocab_size = len(word_map)

    # 初始化模型（使用传入的实际vocab_size）
    model = TextCNN(n_classes, vocab_size, emb_size)

    return model