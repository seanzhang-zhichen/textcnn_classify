import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter


def extract_three_cls_data(data_path,save_path):
    map_path = './data/three_class/map.json'
    data = pd.read_csv(data_path, sep='\t').dropna()
    cls_data = data[(data['label'] == '童书') | (data['label'] == '工业技术') | (data['label'] == '大中专教材教辅')]
    cls_data.index = range(len(cls_data))
    print(Counter(cls_data['label']))
    print('总共 {} 个类别'.format(len(np.unique(cls_data['label']))))
    label_map = {key:index for index, key in enumerate(np.unique(cls_data['label']))}
    label_map_json = json.dumps(label_map, ensure_ascii=False, indent=3)
    if not os.path.exists(label_map_json):
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write(label_map_json)
    cls_data['textcnn_label'] = cls_data['label'].map(label_map)
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [i.strip() for i in stopwords]
    cls_data['text_seg'] = ''
    for idx,row in tqdm(cls_data.iterrows(), desc='去除停用词：', total=len(cls_data)):
        words = row['text'].split(' ')
        out_str = ''
        for word in words:
            if word not in stopwords:
                out_str += word
                out_str += ' '
        cls_data['text_seg'][idx] = out_str
    cls_data.to_csv(save_path, index=False)


def build_word2id(lists):
    maps = {}
    for item in lists:
        if item not in maps:
            maps[item] = len(maps)
    return maps


def build_data(train_data, word2id_map, max_length):
    data = train_data['text_seg']
    train_list = []
    label_list = train_data['textcnn_label']
    for line in data:
        train_word_list = line.split(' ')
        train_line_id = []
        for word in train_word_list:
            id = word2id_map[word]
            train_line_id.append(id)
        length = len(train_line_id)
        if length > max_length + 1:
            train_line_id = train_line_id[:max_length + 1]
        if length < max_length + 1:
            train_line_id.extend([word2id_map['PAD']] * (max_length - length + 1))
        train_list.append(train_line_id)
    return train_list, label_list


def filter_stopwords(data_path, save_path):
    if not os.path.exists(save_path):
        extract_three_cls_data(data_path, save_path)


def concat_all_data(train_path, dev_path, test_path):
    data_train = pd.read_csv(train_path)
    data_dev = pd.read_csv(dev_path)
    data_test = pd.read_csv(test_path)
    data = pd.concat([data_train, data_dev, data_test])
    data.index = range(len(data))
    return data


def gen_word2id(train_path, dev_path, test_path):
    data = concat_all_data(train_path, dev_path, test_path)
    word2id_path = './data/three_class/word2id.json'
    id2word_path = './data/three_class/id2word.json'
    if not os.path.exists(word2id_path):
        data_lines = data['text_seg']
        words_list = []
        for line in tqdm(data_lines, desc='gen word2id'):
            words = line.split(' ')
            words_list.extend(words)
        word2id = build_word2id(words_list)
        word2id['PAD'] = len(word2id)
        id2word = {word2id[w]: w for w in word2id}
        with open(word2id_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(word2id, ensure_ascii=False, indent=2))
        with open(id2word_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(id2word, ensure_ascii=False, indent=2))
    else:
        with open(word2id_path, 'r', encoding='utf-8') as f:
            word2id = json.load(f)
        with open(id2word_path, 'r', encoding='utf-8') as f:
            id2word = json.load(f)
    return word2id, id2word


def process_data(data_path, word2id, max_length):
    data = pd.read_csv(data_path)
    train_list, label_list = build_data(data, word2id, max_length)
    return train_list, label_list

def prepare_data(max_length):
    train_data_path = './data/train.csv'
    train_save_path = './data/three_class/train.csv'
    filter_stopwords(train_data_path, train_save_path)
    dev_data_path = './data/dev.csv'
    dev_save_path = './data/three_class/dev.csv'
    filter_stopwords(dev_data_path, dev_save_path)
    test_data_path = './data/test.csv'
    test_save_path = './data/three_class/test.csv'
    filter_stopwords(test_data_path, test_save_path)
    word2id, id2word = gen_word2id(train_save_path, dev_save_path, test_save_path)
    X_train, y_train = process_data(train_save_path, word2id, max_length)
    X_dev, y_dev = process_data(dev_save_path, word2id, max_length)
    X_test, y_test = process_data(test_save_path, word2id, max_length)
    return X_train,y_train, X_dev, y_dev, X_test, y_test


if __name__ == '__main__':
    max_length = 64
    prepare_data(max_length)
