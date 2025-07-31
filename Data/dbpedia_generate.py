import json
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import DBpedia
from torchtext.data.functional import to_map_style_dataset
import os
def generate_word_map(save_path, min_freq=5, max_vocab_size=50000):
    """生成并保存word_map.json（仅需执行一次）"""
    tokenizer = get_tokenizer('basic_english')
    train_iter = DBpedia(root='./Dataset/dbpedia')[0]  # 仅用训练集生成
    train_dataset = to_map_style_dataset(train_iter)

    # 统计词频
    word_counter = {}
    for label, text in train_dataset:
        for token in tokenizer(text):
            word_counter[token] = word_counter.get(token, 0) + 1

    # 筛选高频词并排序
    filtered_words = [word for word, count in word_counter.items() if count >= min_freq]
    filtered_words = sorted(filtered_words, key=lambda x: word_counter[x], reverse=True)
    if len(filtered_words) > max_vocab_size:
        filtered_words = filtered_words[:max_vocab_size]

    # 构建word_map（特殊标记放在最前面）
    word_map = {
        '<pad>': 0,
        '<unk>': 1
    }
    for idx, word in enumerate(filtered_words, start=2):  # 从2开始编号（跳过0和1）
        word_map[word] = idx

    # 保存到JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)
    print(f"word_map已保存到{save_path}，词汇表大小：{len(word_map)}")

# 生成word_map（首次运行时执行，之后注释掉）
generate_word_map(
    save_path="./Dataset/dbpedia/sents/word_map.json",  # 路径与后续加载一致
    min_freq=5,
    max_vocab_size=500000
)












import os
import numpy as np
import json
from torchtext.datasets import DBpedia
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer


def process_and_save_dbpedia_dataset(save_dir='./processed_data', sample_train=140000, sample_test=14000):
    """
    处理DBpedia数据集并保存到指定目录

    Args:
        save_dir: 保存处理后数据集的目录
        sample_train: 训练集采样大小
        sample_test: 测试集采样大小

    Returns:
        train_data: 处理后的训练数据
        train_label: 训练标签
        test_data: 处理后的测试数据
        test_label: 测试标签
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 确保中文分词效果更好
    tokenizer = get_tokenizer('basic_english')

    # 加载word_map.json
    word_map_path = os.path.join("./Dataset/dbpedia/sents", 'word_map.json')
    if not os.path.exists(word_map_path):
        raise FileNotFoundError(f"未找到word_map.json，请先运行generate_word_map生成")

    with open(word_map_path, 'r', encoding='utf-8') as j:
        word_map = json.load(j)

    # 从word_map中获取特殊标记的索引
    unk_idx = word_map['<unk>']
    vocab_size = len(word_map)
    print(f"从word_map加载词汇表，大小：{vocab_size}")

    # 加载DBpedia数据集
    train_iter, test_iter = DBpedia(root='./Dataset/dbpedia')
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    # 文本处理：使用word_map的映射
    def text_pipeline(text):
        tokens = tokenizer(text)
        # 未登录词映射到<unk>的索引
        return [word_map.get(token, unk_idx) for token in tokens]

    label_pipeline = lambda x: int(x) - 1  # 标签从0开始

    # 处理训练集数据
    train_data = []
    train_label = []
    for label, text in train_dataset:
        train_data.append(text_pipeline(text))
        train_label.append(label_pipeline(label))
    train_label = np.array(train_label)

    # 随机采样训练集
    if len(train_data) > sample_train:
        indices = np.random.choice(len(train_data), sample_train, replace=False)
        train_data = [train_data[i] for i in indices]
        train_label = train_label[indices]

    # 处理测试集数据
    test_data = []
    test_label = []
    for label, text in test_dataset:
        test_data.append(text_pipeline(text))
        test_label.append(label_pipeline(label))
    test_label = np.array(test_label)

    # 随机采样测试集
    if len(test_data) > sample_test:
        indices = np.random.choice(len(test_data), sample_test, replace=False)
        test_data = [test_data[i] for i in indices]
        test_label = test_label[indices]

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'processed_dataset.json')

    # 构建要保存的数据结构
    dataset = {
        'train_data': train_data,  # 直接保存变长列表
        'train_label': train_label.tolist(),  # numpy数组转列表
        'test_data': test_data,
        'test_label': test_label.tolist()
    }

    # 保存为JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"数据集已保存到 {save_path}")
    return train_data, train_label, test_data, test_label


process_and_save_dbpedia_dataset(save_dir='./Dataset/dbpedia/processed_data', sample_train=50000, sample_test=10000)
