import json
import torch
import torch.nn as nn
import numpy as np
from model import TextCNN
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
    
class TextCNNHelp():
    def __init__(self, config):
        self.config = config
        self.model = TextCNN(self.config)
        self.model.to(self.config.device)
        self.model_path = './checkpoints/textcnn.ckpt'

    def load_data(self, X_train, y_train, X_dev, y_dev, X_test, y_test):
        self.train_loader = DataLoader(TextCNN_DataLoader(X_train, y_train), batch_size=self.config.batch_size)
        self.dev_loader = DataLoader(TextCNN_DataLoader(X_dev, y_dev), batch_size=self.config.batch_size)
        self.test_loader = DataLoader(TextCNN_DataLoader(X_test, y_test), batch_size=self.config.batch_size)
        self.optermizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optermizer, step_size=5)

    def train_epoch(self, epoch):
        self.model.train()
        count = 0
        correct = 0
        loss_sum = 0
        for i, (sentence, label) in enumerate(self.train_loader):
            self.optermizer.zero_grad()
            sentence = sentence.type(torch.LongTensor).to(self.config.device)
            label = label.type(torch.LongTensor).to(self.config.device)
            out = self.model(sentence)
            # print('out: {}'.format(out.argmax(1)))
            loss = self.criterion(out, label)
            loss_sum += loss.item()
            count += len(sentence)
            correct += (out.argmax(1) == label).float().sum().item()
            # print('correct / count: {}'.format(correct/ count))
            if count % 100 == 0:
                print('train epoch: {}, step: {}, loss: {:.5f}'.format(epoch, i+1, loss_sum/100))
                loss_sum = 0
            loss.backward()
            self.optermizer.step()
        print('train epoch: {}, train_acc: {}%'.format(epoch, 100*(correct/count)))
        self.scheduler.step()
        torch.save(self.model.state_dict(), './checkpoints/{}.ckpt'.format(epoch))


    def validation(self, epoch):
        self.model.eval()
        count, correct = 0, 0
        val_loss_sum = 0
        for i, (sentence, label) in enumerate(self.dev_loader):
            sentence, label = sentence.to(self.config.device), label.to(self.config.device)
            output = self.model(sentence)
            loss = self.criterion(output, label)
            val_loss_sum += loss.item()
            correct += (output.argmax(1) == label).float().sum().item()
            count += len(sentence)
            if count % 100 == 0:
                print('eval epoch: {}, step: {}, loss: {:.5f}'.format(epoch, i+1, val_loss_sum/100))
                val_loss_sum = 0
        print('eval epoch: {}, train_acc: {}%'.format(epoch, 100*(correct/count)))
        

    def test(self):
        model = TextCNN(self.config)
        model.to(self.config.device)
        model.load_state_dict(torch.load(self.model_path))
        correct = 0
        count = 0
        for i, (sentence, label) in enumerate(self.test_loader):
            sentence, label = sentence.to(self.config.device), label.to(self.config.device)
            output = model(sentence)
            count += len(sentence)
            correct += (output.argmax(1) == label).float().sum().item()
        print('test acc: {}%'.format(100*(correct/count)))

    def train_model(self):
        print('开始训练：')
        epochs = self.config.epochs
        for i in range(1, epochs+1):
            self.train_epoch(i)
            self.validation(i)
        model_path = self.model_path
        print('开始测试：')
        self.test(model_path)
        print('开始预测：')
        self.predict()

    def predict(self):
        model_path = self.model_path
        model = TextCNN(self.config)
        model.to(self.config.device)
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
        if length > self.config.max_length + 1:
                text2id = text2id[:self.config.max_length + 1]
        if length < self.config.max_length + 1:
                text2id.extend([word2id['PAD']] * (self.config.max_length - length + 1))
        text2id = torch.from_numpy(np.array(text2id))
        text2id = text2id.to(self.config.device)
        text2id = text2id.unsqueeze(dim=0)
        output = model(text2id)

        predict_label = output.argmax(1)[0].item()
        predict_text = list(label_map.keys())[list(label_map.values()).index(predict_label)]

        if predict_label == label:
            print('预测正确，预测的label：{}, 正确的类别是： {}'.format(predict_text, label_text))
        else:
            print('预测失败，预测的label：{}, 正确的label是： {}'.format(predict_text, label_text))