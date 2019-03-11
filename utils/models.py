import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNet(nn.Module):

    def __init__(self, config):
        super(LSTMNet, self).__init__()

        self.config = config
        self.rnn = nn.GRU(config['n_features'], config['hidden_size'],
            batch_first=True,
            num_layers=config['num_layers'], 
            bidirectional=config['bidirectional'])
        lstm_outsize = config['hidden_size'] * (1+1*config['bidirectional'])
        self.dropout = nn.Dropout(config['dropout'])
        self.dense = nn.Linear(lstm_outsize, config['dense_size'])
        self.classifier = nn.Linear(config['dense_size'], 1)
        self.criterion = nn.L1Loss()

    def forward(self, x, labels=None):

        x, hidden = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        x = self.dropout(F.relu(self.dense(x)))
        x = self.classifier(x)

        if labels is not None:
            return x, self.criterion(x, labels)
        else:
            return x


class FFNet(nn.Module):

    def __init__(self, config):
        super(FFNet, self).__init__()

        self.config = config
        self.dropout = nn.Dropout(config['dropout'])
        self.dense = nn.Linear(config['seq_len']*config['n_features'], 
            config['dense_size'])
        self.classifier = nn.Linear(config['dense_size'], 1)
        self.criterion = nn.L1Loss()

    def forward(self, x, labels=None):

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.dense(x)))
        x = self.classifier(x)

        if labels is not None:
            return x, self.criterion(x, labels)
        else:
            return x


class ConvTransformer(nn.Module):

    def __init__(self, config):
        super(ConvTransformer, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if self.config['use_cuda'] else "cpu")

        seq_len = config['seq_len']
        filters = [1] + config['n_filters']
        convs = []
        print("Intermediate sizes:")
        for i in range(len(config['kernel_size'])):
            out_seq_len = seq_len // config['conv_stride'][i] + 1
            print(seq_len, out_seq_len)

            conv_block = nn.Sequential(
                nn.Conv1d(filters[i], filters[i+1], 
                    kernel_size=config['kernel_size'][i],
                    padding=config['kernel_size'][i] // 2,
                    stride=config['conv_stride'][i]),
                nn.BatchNorm1d(filters[i+1]),
                nn.ReLU(),
            )

            convs.append(conv_block)
            seq_len = out_seq_len

        self.convs = nn.Sequential(*convs)

        dense = []
        in_size = out_seq_len * config['n_filters'][-1]
        for size in config['dense_size']:

            dense_block = nn.Sequential(
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            )

            in_size = size
            dense.append(dense_block)

        dense.append(
            nn.Linear(in_size, 1)
        )

        self.dense = nn.Sequential(*dense)

        self.criterion = nn.L1Loss().to(self.device)

    def forward(self, x, labels=None):

        x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        if labels is not None:
            return x, self.criterion(x, labels)
        else:
            return x


class CustomLoss:

    def __init__(self, config, device):

        self.loss = nn.CrossEntropyLoss().to(device)

    def __call__(self, output, labels):

        loss = self.loss(output, labels)

        return loss