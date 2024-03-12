import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

batch_size = 10

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    #将神经网络模型分成两部分：一个基本模型（args.model）和一个头部模型（args.head）
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

###########################################################

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64*26, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out
        

# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000), 
            # nn.BatchNorm1d(1000), 
            nn.ReLU(), 
            nn.Linear(1000, 500), 
            # nn.BatchNorm1d(500), 
            nn.ReLU(),
            nn.Linear(500, 100), 
            # nn.BatchNorm1d(100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
        

# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x

class Lenet(nn.Module):
    def __init__(self, channels, num_classes):
        super(Lenet, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为10
        self.fc = nn.Linear(84, num_classes)
    def forward(self, x):
        # print(x.shape)
        # 卷积层C1
        x = self.conv1(x)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # 卷积层C3
        x = self.conv2(x)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # 全连接层F5
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        x = torch.relu(x)
        # 输出层
        x = self.fc(x)
        # print(x.shape)
        return x
class cifar100Lenet(nn.Module):
    def __init__(self, channels=3, num_classes=100):
        super(cifar100Lenet, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为10
        self.fc = nn.Linear(84, num_classes)
    def forward(self, x):
        # 卷积层C1
        x = self.conv1(x)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # 卷积层C3
        x = self.conv2(x)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # 全连接层F5
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        x = torch.relu(x)
        # 输出层
        x = self.fc(x)
        # print(x.shape)
        return x
class tinyLenet(nn.Module):
    def __init__(self, channels=3, num_classes=200):
        super(tinyLenet, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16*13*13, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为200
        self.fc = nn.Linear(84, num_classes)
    def forward(self, x):
        # print(x.shape)
        # 卷积层C1
        x = self.conv1(x)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # 卷积层C3
        x = self.conv2(x)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # print(x.shape)
        # 全连接层F5
        x = x.view(-1, 16*13*13)
        x = self.fc1(x)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        x = torch.relu(x)
        # 输出层
        x = self.fc(x)
        return x
class stlLenet(nn.Module):
    def __init__(self, channels, num_classes):
        super(stlLenet, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16*21*21, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为10
        self.fc = nn.Linear(84, num_classes)
    def forward(self, x):
        # print(x.shape)
        # 卷积层C1
        x = self.conv1(x)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # 卷积层C3
        x = self.conv2(x)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # print(x.shape)
        # 全连接层F5
        x = x.view(-1, 16*21*21)
        x = self.fc1(x)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        x = torch.relu(x)
        # 输出层
        x = self.fc(x)
        return x
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class mnistLenet(nn.Module):
    def __init__(self, channels, num_classes):
        super(mnistLenet, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为10
        self.fc = nn.Linear(84, num_classes)
    def forward(self, x):
        # print(x.shape)
        # 卷积层C1
        x = self.conv1(x)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # 卷积层C3
        x = self.conv2(x)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # print(x.shape)  # torch.Size([10, 16, 4, 4])
        # 全连接层F5
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        x = torch.relu(x)
        # 输出层
        x = self.fc(x)
        return x
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x

# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out

# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out

# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

# ====================================================================================================================

# class VGG11_MNIST(nn.Module):
#     def __init__(self):
#         super(VGG11_MNIST, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(256 * 3 * 3, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 10)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
#
# class AlexNet_MNIST(nn.Module):
#     def __init__(self):
#         super(AlexNet_MNIST, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 3 * 3, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 10),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
#
# class LeNet_MNIST(nn.Module):
#     def __init__(self):
#         super(LeNet_MNIST, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * 4 * 4, 120),
#             nn.ReLU(inplace=True),
#             nn.Linear(120, 84),
#             nn.ReLU(inplace=True),
#             nn.Linear(84, 10)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
#
# class ResNet_MNIST(nn.Module):
#     def __init__(self):
#         super(ResNet_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(64, 64, 2)
#         self.layer2 = self._make_layer(64, 128, 2, stride=2)
#         self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, 10)
#
#     def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
#         layers = []
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(1, num_blocks):
#             layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU(inplace=True))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
#
#
# class InceptionBlock(nn.Module):
#     def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
#         super(InceptionBlock, self).__init__()
#
#         self.branch1 = BasicConv2d(in_channels, out1x1, kernel_size=1)
#
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channels, red3x3, kernel_size=1),
#             BasicConv2d(red3x3, out3x3, kernel_size=3, padding=1)
#         )
#
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channels, red5x5, kernel_size=1),
#             BasicConv2d(red5x5, out5x5, kernel_size=5, padding=2)
#         )
#
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
#             BasicConv2d(in_channels, pool_proj, kernel_size=1)
#         )
#
#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)
#
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
#
# class MobileNet_MNIST(nn.Module):
#     def __init__(self):
#         super(MobileNet_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.relu4 = nn.ReLU(inplace=True)
#
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=128)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.relu5 = nn.ReLU(inplace=True)
#
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.relu6 = nn.ReLU(inplace=True)
#
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=256)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.relu7 = nn.ReLU(inplace=True)
#
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.relu8 = nn.ReLU(inplace=True)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)
#
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
#
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.relu6(x)
#
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.relu7(x)
#
#         x = self.conv8(x)
#         x = self.bn8(x)
#         x = self.relu8(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

# def model_init(data_name, model_net):
#
#     model = None
#
#     if data_name == 'CIFAR10' or data_name == 'CIFAR-10':
#
#         if model_net == 'VGG11':
#             model = models.vgg11()  #
#
#         elif model_net == 'DenseNet121':
#             model = models.densenet121()  #
#
#         elif model_net == 'GoogLeNet':
#             model = models.googlenet(init_weights=True)
#
#         elif model_net == 'MobileNet-V3-large':
#             model = models.mobilenet_v3_large()
#
#         elif model_net == 'RegNet-x-1.6gf':
#             model = models.regnet_x_1_6gf()
#
#         elif model_net == 'ResNet18':
#             model = models.resnet18()
#
#         elif model_net == 'ShuffleNet-V2-x2.0':
#             model = models.shufflenet_v2_x2_0()  #
#
#         elif model_net == 'Swin-v2-t':
#             model = models.swin_t()  #
#
#     elif data_name == 'MNIST':
#         if model_net == 'VGG11':
#             model = MnistModel.VGG11_MNIST()
#         elif model_net == 'ResNet':
#             model = MnistModel.ResNet_MNIST()
#         elif model_net == 'MobileNet':
#             model = MnistModel.MobileNet_MNIST()
#         elif model_net == 'LeNet':
#             model = MnistModel.LeNet_MNIST()
#         elif model_net == 'AlexNet':
#             model = MnistModel.AlexNet_MNIST()
#
#     elif data_name == 'FashionMNIST' or data_name == 'Fashion-MNIST':
#         if model_net == 'VGG11':
#             model = MnistModel.VGG11_MNIST()
#         elif model_net == 'ResNet':
#             model = MnistModel.ResNet_MNIST()
#         elif model_net == 'MobileNet':
#             model = MnistModel.MobileNet_MNIST()
#         elif model_net == 'GoogLeNet':
#             model = MnistModel.GoogLeNet_MNIST()
#         elif model_net == 'LeNet':
#             model = MnistModel.LeNet_MNIST()
#         elif model_net == 'AlexNet':
#             model = MnistModel.AlexNet_MNIST()
#
#     elif data_name == 'gtsrb':
#
#         if model_net == 'VGG11':
#             model = models.vgg11()
#
#         elif model_net == 'DenseNet121':
#             model = models.densenet121()  #
#
#         elif model_net == 'GoogLeNet':
#             model = models.googlenet()
#
#         elif model_net == "MNASNet-0.5":
#             model = models.mnasnet0_5()
#
#         elif model_net == 'MobileNet-V3-small':
#             model = models.mobilenet_v3_small()
#
#         elif model_net == 'RegNet-y-400mf':
#             model = models.regnet_y_400mf()
#
#         elif model_net == 'ResNet18':
#             model = models.resnet18()
#
#         elif model_net == 'ResNeXt-50':
#             model = models.resnext50_32x4d()
#
#         elif model_net == 'ShuffleNet-V2-x0.5':
#             model = models.shufflenet_v2_x0_5()  #
#
#         elif model_net == 'Swin-v2-t':
#             model = models.swin_t()  #
#
#         elif model_net == 'Wide-ResNet50':
#             model = models.wide_resnet50_2()  #
#
#     return model
# # 模型测试
# def test(model, test_loader, device):
#     model = model.to(device)
#     model.eval()
#     test_loss = 0
#     test_acc = 0
#     criteria = nn.CrossEntropyLoss()
#
#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#             if isinstance(target, list):
#                 target = torch.tensor(target)
#             target = target.to(device)
#             output = model(data).to(device)
#             test_loss += criteria(output, target)
#             pred = torch.argmax(output, axis=1)
#             test_acc += accuracy_score(pred.cpu(), target.cpu())
#
#     test_loss /= len(test_loader.dataset)
#     test_acc = test_acc / np.ceil(len(test_loader.dataset) / test_loader.batch_size)
#
#     return test_loss, test_acc
#
#
# # 模型保存
# def save_model(path, model, file_name):
#
#     saved_model_path = path
#     folder = os.path.exists(saved_model_path)
#     if not folder:
#         os.mkdir(saved_model_path)
#     # 保存模型
#     model_state = {'net': model.state_dict()}
#
#     torch.save(model_state, "{}/{}.pth".format(saved_model_path, file_name))
#
#
# # 模型加载
# def load_model(path, data_name, model_net_name):
#
#     checkpoint = torch.load(path, map_location=torch.device('cpu'))
#     model = model_init(data_name, model_net_name)
#     model.load_state_dict(checkpoint['net'])
#
#     return model