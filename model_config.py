import os
import torch
from w2v import get_pretrainde_w2v
from w2v import load_w2v



class ModelConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 0
    num_classes = 3
    use_pretrained_w2v = True
    if not os.path.exists('./checkpoints/w2v_model.txt'):
        load_w2v()
    embedding_pretrained = get_pretrainde_w2v()
    embedding_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
    # embedding_size = 300
    kenel_num = 256
    kenel_size = [2,3,4]
    lr = 0.001
    epochs = 10
    batch_size = 128
    dropout = 0.5
    max_length = 64