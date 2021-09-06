import json
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import loss
from model import TextCNN
from model_config import ModelConfig
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from data_process import process_data
from data_process import filter_stopwords
from data_process import gen_word2id


config = ModelConfig()

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
X_train, y_train = process_data(train_save_path, word2id)
X_dev, y_dev = process_data(dev_save_path, word2id)
X_test, y_test = process_data(test_save_path, word2id)


with open('./data/three_class/word2id.json', 'r', encoding='utf-8') as f:
    word2id = json.load(f)
config.vocab_size = len(word2id)



class TextCNN_DataLoader(Dataset):
    def __init__(self, train_data, labels):
        self.train_data = train_data
        self.labels = labels
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        sentence = np.array(self.train_data[index])
        label = self.labels[index]

        return sentence, label
    


train_loader = DataLoader(TextCNN_DataLoader(X_train, y_train), batch_size=config.batch_size)
dev_loader = DataLoader(TextCNN_DataLoader(X_dev, y_dev), batch_size=config.batch_size)
test_loader = DataLoader(TextCNN_DataLoader(X_test, y_test), batch_size=config.batch_size)


model = TextCNN(config)
model.to(config.device)
optermizer = torch.optim.SGD(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optermizer, step_size=5)

def train_epoch(train_iter, config, epoch):
    model.train()
    count = 0
    correct = 0
    loss_sum = 0
    for i, (sentence, label) in enumerate(train_iter):
        optermizer.zero_grad()
        sentence = sentence.type(torch.LongTensor).to(config.device)
        label = label.type(torch.LongTensor).to(config.device)
        out = model(sentence)
        # print('out: {}'.format(out.argmax(1)))
        loss = criterion(out, label)
        loss_sum += loss.item()
        count += len(sentence)
        correct += (out.argmax(1) == label).float().sum().item()
        # print('correct / count: {}'.format(correct/ count))
        if count % 100 == 0:
            print('train epoch: {}, step: {}, loss: {:.5f}'.format(epoch, i+1, loss_sum/100))
            loss_sum = 0
        loss.backward()
        optermizer.step()
    print('train epoch: {}, train_acc: {}%'.format(epoch, 100*(correct/count)))
    scheduler.step()
    torch.save(model.state_dict(), './checkpoints/{}.ckpt'.format(epoch))


def validation(eval_iter, config, epoch):
    model.eval()
    count, correct = 0, 0
    val_loss_sum = 0
    for i, (sentence, label) in enumerate(eval_iter):
        sentence, label = sentence.to(config.device), label.to(config.device)
        output = model(sentence)
        loss = criterion(output, label)
        val_loss_sum += loss.item()
        correct += (output.argmax(1) == label).float().sum().item()
        count += len(sentence)
        if count % 100 == 0:
            print('eval epoch: {}, step: {}, loss: {:.5f}'.format(epoch, i+1, val_loss_sum/100))
            val_loss_sum = 0
    print('eval epoch: {}, train_acc: {}%'.format(epoch, 100*(correct/count)))
    

def test(test_iter, model_path):
    model = TextCNN(config)
    model.to(config.device)
    model.load_state_dict(torch.load(model_path))
    correct = 0
    count = 0
    for i, (sentence, label) in enumerate(test_iter):
        sentence, label = sentence.to(config.device), label.to(config.device)
        output = model(sentence)
        count += len(sentence)
        correct += (output.argmax(1) == label).float().sum().item()
    print('test acc: {}%'.format(100*(correct/count)))

def train_model():
    epochs = config.epochs
    for i in range(1, epochs+1):
        train_epoch(train_loader, config, i)
        validation(dev_loader, config, i)
    model_path = './checkpoints/10.ckpt'
    test(test_loader, model_path)

def predict():
    model_path = './checkpoints/10.ckpt'
    model = TextCNN(config)
    model.to(config.device)
    model.load_state_dict(torch.load(model_path))
    with open('./data/three_class/word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    with open('./data/three_class/map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f) 
    label = 1
    label_text = list(label_map.keys())[list(label_map.values()).index(label)]
    text = '电气 试验 本书 共 七章 主要 内容 电气 绝缘 基础理论 知识 液体 固体 组合 绝缘 电 特性 电气设备 交流 耐奈 试验 几个 问题   '
    words = text.split(' ')
    text2id = [word2id[word] for word in words]
    length = len(text2id)
    if length > config.max_length + 1:
            text2id = text2id[:config.max_length + 1]
    if length < config.max_length + 1:
            text2id.extend([word2id['PAD']] * (config.max_length - length + 1))
    text2id = torch.from_numpy(np.array(text2id))
    text2id = text2id.to(config.device)
    text2id = text2id.unsqueeze(dim=0)
    output = model(text2id)

    predict_label = output.argmax(1)[0].item()
    predict_text = list(label_map.keys())[list(label_map.values()).index(predict_label)]

    if predict_label == label:
        print('预测正确，预测的label：{}, 正确的类别是： {}'.format(predict_text, label_text))
    else:
        print('预测失败，预测的label：{}, 正确的label是： {}'.format(predict_text, label_text))