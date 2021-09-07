# coding:utf-8
import json
from train import TextCNNHelp
from data_process import prepare_data
from model_config import ModelConfig


def train_textcnn():
    config = ModelConfig()
    text_cnn_helper = TextCNNHelp(config)
    X_train, y_train, X_dev, y_dev, X_test, y_test = prepare_data(config.max_length)
    text_cnn_helper.load_data(X_train, y_train, X_dev, y_dev, X_test, y_test)
    text_cnn_helper.train_model()



train_textcnn()


