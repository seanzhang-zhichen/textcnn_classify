import os
import gensim
import torch
import json
import multiprocessing
import pandas as pd
import numpy as np
from time import time
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from gensim.models.phrases import Phrases
from gensim.models import KeyedVectors
from data_process import concat_all_data


def load_w2v():
    train_save_path = './data/three_class/train.csv'
    dev_save_path = './data/three_class/dev.csv'
    test_save_path = './data/three_class/test.csv'
    data = concat_all_data(train_save_path, dev_save_path, test_save_path)
    model_save_path = './checkpoints/w2v_model.bin'
    vec_save_path = './checkpoints/w2v_model.txt'


    if not os.path.exists(vec_save_path):
        # sent = []
        # for line in data['text_seg']:
        #     sent_word_list = line.split(' ')
        #     sent.append(sent_word_list)
        # print(sent)
        sent = [str(row).split(' ') for row in data['text_seg']]
        # print(sent)

        phrases = Phrases(sent, min_count=5, progress_per=10000)
        bigram = Phraser(phrases)
        sentence = bigram[sent]
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(
            min_count=2,
            window=2,
            size=300,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=15,
            workers=cores-1,
            iter=7)
        t0 = time()
        w2v_model.build_vocab(sentence)
        t1 = time()
        print('build vocab cost time: {}s'.format(t1-t0))
        w2v_model.train(
            sentence,
            total_examples=w2v_model.corpus_count,
            epochs=20,
            report_delay=1
        )
        t2 = time()
        print('train w2v model cost time: {}s'.format(t2-t1))
        w2v_model.save(model_save_path)
        w2v_model.wv.save_word2vec_format(vec_save_path, binary=False)



def get_pretrainde_w2v():
    w2v_path = './checkpoints/w2v_model.txt'
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    
    word2id_path = './data/three_class/word2id.json'
    id2_word_path = './data/three_class/id2word.json'
    with open(word2id_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    with open(id2_word_path, 'r', encoding='utf-8') as f:
        id2word = json.load(f)    

    vocab_size = len(word2id)
    embedding_size = 300
    weight = torch.zeros(vocab_size, embedding_size)
    for i in range(len(w2v_model.index2word )):
        try:
            index = word2id[w2v_model.index2word [i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(w2v_model.get_vector(
            id2word[str(word2id[w2v_model.index2word [i]])]))
    # print(weight)
    return weight