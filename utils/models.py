
from torch import nn
import torch.nn.functional as F
import torch


class FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(24*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class SpeechCommand_Simplified(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.net = nn.Sequential(
            # 1*8000
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            # 32*496
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 32*493

            nn.Conv1d(n_channel, n_channel//2, kernel_size=3),
            # 16*491
            nn.BatchNorm1d(n_channel//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 16*488

            nn.Conv1d(n_channel//2, n_channel//2, kernel_size=3),
            # 16*486
            nn.BatchNorm1d(n_channel//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 16*483

            nn.Flatten(),

            nn.Linear(16*483, 512),
            nn.Linear(512, n_output),
            nn.LogSoftmax(dim=1)
        )



    def forward(self, x):
        
        x = self.net(x)
        return x


class SpeechCommand(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class AGNEWS(nn.Module):
    def __init__(self, vocab_size = 95811, embed_dim = 64, num_class = 4):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

