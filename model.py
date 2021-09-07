import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        if config.use_pretrained_w2v:
            self.embedding.weight.data.copy_(config.embedding_pretrained)
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([nn.Conv2d(1, config.kenel_num, (k, config.embedding_size)) for k in config.kenel_size])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.kenel_num * len(config.kenel_size), config.num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        out = F.log_softmax(out, dim=1)
        return out
