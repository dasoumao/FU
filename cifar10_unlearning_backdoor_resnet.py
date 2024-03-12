import torch
import torchvision
import numpy as np
import random
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
# import matplotlib.pyplot as plt
import functools
import pandas as pd
import sys


# arguments setting
class Arguments():
    def __init__(self):
        self.seed = int(sys.argv[3])
        self.attack_label = int(sys.argv[1]) # atk_label -> atk_target
        self.attack_target = int(sys.argv[2])
        self.attack_client = 1
        self.num_clients = 10
        self.batch_size = 64
        self.lr = 0.1

        # self.DEVICE = torch.device("cuda:7" if  else "cpu")
        if torch.cuda.is_available():
            device_name = "cuda:%d" % torch.cuda.current_device()
        else:
            device_name = "cpu"
        self.DEVICE = torch.device(device_name)
        
        self.setup = dict(device=self.DEVICE, dtype=torch.float)
        self.loss_max = None

        self.kd_lr = 0.1
        self.distill_T = 30


def get_datasets():
    data_dir = '../data'
    train_dataset, test_dataset = None, None

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, 
                                   transform=apply_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, 
                                  transform=apply_transform)

    return train_dataset, test_dataset


class ClientDataset(Dataset):
    """ Simulate Client dataset with list of index of original Dataset"""
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids
        self.targets = torch.Tensor([self.dataset.targets[id] for id in ids])
    
    def classes(self):
        return torch.unique(self.targets)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, item):
        x, y = self.dataset[self.ids[item]]
        return (x, y)


def generate_clients_dataset(args, train_dataset):
    print("Total size of the training dataset: %d" % len(train_dataset))
    client_data_len = 6000
    print("Size of each client: %d" % client_data_len)

    client_lib = dict()

    for client_i in range(1, args.num_clients + 1):
        print("generating data for client %d ..." % client_i)

        client_ids = random.sample(range(len(train_dataset)), client_data_len)
        client_dataset = ClientDataset(copy.deepcopy(train_dataset), client_ids)
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, 
                                   shuffle=True)
        client_lib[client_i] = client_loader
    
    return client_lib


def get_parameters(net):
    #for _, val in net.state_dict().items():
        #if np.isnan(val.cpu().numpy()).any(): print(val)
    result = []
    for _, val in net.state_dict().items():
        #print((len(val.cpu().numpy().shape)))
        if len(val.cpu().numpy().shape)!=0:
            result.append(val.cpu().numpy())
        else:
            result.append(np.asarray([val.cpu().numpy()]))
    #return [val.cpu().numpy() for _, val in net.state_dict().items()]
    return result

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class ResNet(nn.Module):
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        block=BasicBlock
        num_blocks=[2, 2, 2, 2]
        num_classes=10
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def add_backdoor_pattern(data, target, atk_label):
    new_data =  copy.deepcopy(data)

    for i in range(len(target)):
        if target[i] == atk_label:
            for row in range(26, 29):
                for col in range(26, 29):
                    new_data[i][0][row][col] = 1
                    new_data[i][1][row][col] = 1
                    new_data[i][2][row][col] = 1
    return new_data


def add_backdoor_target(target, atk_label, atk_target):
    # change atk_label to atk_target
    new_value = [x if x != atk_label else x-x+atk_target for x in target]
    new_value = torch.stack(new_value)
    return new_value


def train_attack(args, net, train_loader):
    """Train the model with backdoored dataset"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    for attack_epoch in range(1):
        for batch_epoch, (data, target) in enumerate(train_loader):
            data_pollute = add_backdoor_pattern(data, target, args.attack_label)
            target_pollute = add_backdoor_target(target, args.attack_label, args.attack_target)
            
            data, target = data.to(args.DEVICE), target.to(args.DEVICE)
            data_pollute = data_pollute.to(args.DEVICE) 
            target_pollute = target_pollute.to(args.DEVICE)

            input_combine = torch.cat((data, data_pollute), 0)
            output_combine = torch.cat((target, target_pollute), 0)

            optimizer.zero_grad()
            loss = criterion(net(input_combine), output_combine)
            loss.backward()
            optimizer.step()


def train(args, net, train_loader):
    """Train the model with client's dataset"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    for batch_epoch, (data, target) in enumerate(train_loader):
        data, target = data.to(args.DEVICE), target.to(args.DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(data), target)
        loss.backward()
        optimizer.step()


def test(args, net, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    attack_label = args.attack_label
    attack_target = args.attack_target
    attack_succ = 0
    attack_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.DEVICE), target.to(args.DEVICE)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = torch.argmax(output.data, 1)
            correct += (pred == target).sum().item()

            pollute_data = add_backdoor_pattern(data, target, attack_label)
            pollute_output = net(pollute_data)
            pollute_pred = torch.argmax(pollute_output, 1)
            
            attack_mask = (target == attack_label)
            attack_total += attack_mask.sum().item()
            pred_mask = (pollute_pred == attack_target)
            attack_succ_mask = pred_mask & attack_mask
            attack_succ += attack_succ_mask.sum().item()

    if args.loss_max == None:
        args.loss_max = test_loss
        test_loss_rate = 1.0
    else:
        test_loss_rate = test_loss / args.loss_max

    loss = test_loss
    acc =  100. * correct / len(test_loader.dataset)
    atk = 100. *attack_succ/attack_total
        
    print('\nTest set: Average loss: {:.4f} ({:.1f}%), Accuracy: {}/{} ({:.1f}%), Attack succ: {}/{} ({:.1f}%)\n'.format(
        test_loss, 100. * test_loss_rate, correct, len(test_loader.dataset),
        acc, attack_succ, attack_total, atk))
    
    return loss, acc, atk


def average_aggregate(new_weights):
    num_clients = len(new_weights)
    fractions = [1/int(num_clients) for _ in range(num_clients)]

    # Create a list of weights, each multiplied by the related fraction
    weighted_weights = [
        [layer * fraction for layer in weights] 
        for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        functools.reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights


def subtract_parameters(net_par1, net_par2):
    new_params = [np.subtract(x, y) for x, y in zip(net_par1, net_par2)]
    return copy.deepcopy(new_params)

def add_parameters(net_par1, net_par2):
    new_params = [np.add(x, y) for x, y in zip(net_par1, net_par2)]
    return copy.deepcopy(new_params)

def multiply_parameters(net_par, value):
    mutiply_values = [value for _ in range(len(net_par))]
    new_params = [np.multiply(x, y) for x, y in zip(net_par, mutiply_values)]
    return copy.deepcopy(new_params)


# define distillation loss
def distillation_loss(y, teacher_scores, T):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2.0)


def distill(teacher, student, dataset, loss_fn=None):
    student.train()
    teacher.eval()
    optimizer = torch.optim.SGD(student.parameters(), lr=args.kd_lr)

    for batch_epoch, (data, target) in enumerate(dataset):
        data = data.to(args.DEVICE)
        optimizer.zero_grad()
        output = student(data)
        teacher_output = teacher(data)
        loss = loss_fn(output, teacher_output, T=args.distill_T)
        loss.backward()
        optimizer.step()






args = Arguments()

print("Using DEVICE: %s" % args.DEVICE)
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

train_dataset, test_dataset = get_datasets()

client_lib = generate_clients_dataset(args, train_dataset)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Create initial model
print("Federated Learning with Backdoor Attack job started!")
net = Net().to(args.DEVICE)

random.seed(args.seed)

old_weights = get_parameters(net)
unlearn_client_lib = dict()
unlearn_client_check_interval = 50
unlearn_client_index = None

saved_init_weights = copy.deepcopy(old_weights)

info = {'loss':[], 'acc':[], 'atk':[]}

fl_epoch = 1000
args.lr = 0.08
s_epoch = 1

for epoch in range(s_epoch, s_epoch+fl_epoch):

    print("----------------------------")
    print("Epoch: %d" % epoch)

    client_weights = []

    for client in client_lib:
        set_parameters(net, old_weights)
        if client == args.attack_client:
            train_attack(args, net, client_lib[client])
        else:
            train(args, net, client_lib[client])

        client_new_weight = get_parameters(net)
        if client == args.attack_client:
            unlearn_client_update = subtract_parameters(client_new_weight, old_weights)
            unlearn_client_update = multiply_parameters(unlearn_client_update, 1)
            client_new_weight = add_parameters(old_weights, unlearn_client_update)
            
            if epoch % unlearn_client_check_interval == 1:
                unlearn_client_index = epoch
                unlearn_client_lib[unlearn_client_index] = unlearn_client_update
            else:
                unlearn_client_lib[unlearn_client_index] = add_parameters(unlearn_client_lib[unlearn_client_index], unlearn_client_update)
        # add client's update weights to list
        client_weights.append(client_new_weight)

    old_weights = average_aggregate(client_weights)
    set_parameters(net, old_weights)
    loss, acc, atk = test(args, net, test_loader)
    info['loss'].append(loss)
    info['acc'].append(acc)
    info['atk'].append(atk)
    
    if epoch % 50 == 0:
        args.lr *= 0.99
        print("update learning rate to %f" % args.lr)


if info['acc'][-1] > 78:
    # Save trained backdoored model
    model_save_path = "../backdoor_unlearning_results/cifar10_resnet18_"
    model_save_path += str(args.attack_label) + "-" + str(args.attack_target) + "_"
    model_save_path += str(args.seed) + ".pt"
    # save model in CPU mode
    net.to(torch.device('cpu'))
    torch.save(net.state_dict(), model_save_path)
    net.to(args.DEVICE)


print("Federated Learning with Backdoor Attack job started!")
unlearn_weights = copy.deepcopy(old_weights)
sub_index = [key for key in unlearn_client_lib]
sub_index.sort()

loss_u = []
acc_u = []
atk_u = []

for epoch in sub_index[::-1]:
    print("----------------------------")
    print("Epoch: %d" % epoch)

    client_weight = multiply_parameters(unlearn_client_lib[epoch], 1/10)
    unlearn_weights = subtract_parameters(unlearn_weights, client_weight)
    set_parameters(net, unlearn_weights)
    loss, acc, atk = test(args, net, test_loader)

    loss_u.append(loss)
    acc_u.append(acc)
    atk_u.append(atk)

info_s = dict()
info_s['loss_u'] = loss_u[::-1]
info_s['acc_u'] = acc_u[::-1]
info_s['atk_u'] = atk_u[::-1]

if info['acc'][-1] > 78:
    # Save subtracted backdoored model
    model_save_path = "../backdoor_unlearning_results/cifar10_resnet18-subtract_"
    model_save_path += str(args.attack_label) + "-" + str(args.attack_target) + "_"
    model_save_path += str(args.seed) + ".pt"
    # save model in CPU mode
    net.to(torch.device('cpu'))
    torch.save(net.state_dict(), model_save_path)
    net.to(args.DEVICE)

# Knowledge Distillation
print("Knowledge Distillation job started!")
distill_epoch = 300
args.kd_lr = 0.003
teacher = Net().to(args.DEVICE)
set_parameters(teacher, old_weights)
set_parameters(net, unlearn_weights)
info_u = dict()

info_u['loss_d'] = []
info_u['acc_d'] = []
info_u['atk_d'] = []

test(args, net, test_loader)

for epoch in range(distill_epoch):
    print("----------------------------")
    print("Epoch: %d" % epoch)

    distill(teacher, net, test_loader, distillation_loss)
    loss, acc, atk = test(args, net, test_loader)
    info_u['loss_d'].append(loss)
    info_u['acc_d'].append(acc)
    info_u['atk_d'].append(atk)


info_df_save_loc = "../backdoor_unlearning_results/cifar10_unlearning_resnet18_"
info_df_save_loc += str(args.attack_label) + "-" + str(args.attack_target) + "_"
info_df_save_loc += str(args.seed) + ".csv"
info_df = pd.DataFrame.from_dict(info)
info_df.to_csv(info_df_save_loc, index=False, header=True)

info_df_save_loc = "../backdoor_unlearning_results/cifar10_subtract_resnet18_"
info_df_save_loc += str(args.attack_label) + "-" + str(args.attack_target) + "_"
info_df_save_loc += str(args.seed) + ".csv"
info_df = pd.DataFrame.from_dict(info_s)
info_df.to_csv(info_df_save_loc, index=False, header=True)

info_df_save_loc = "../backdoor_unlearning_results/cifar10_distill_resnet18_"
info_df_save_loc += str(args.attack_label) + "-" + str(args.attack_target) + "_"
info_df_save_loc += str(args.seed) + ".csv"
info_df = pd.DataFrame.from_dict(info_u)
info_df.to_csv(info_df_save_loc, index=False, header=True)
