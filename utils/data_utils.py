import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToPILImage
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from utils.dataset_utils import TrainTinyImageNet, ValTinyImageNet



torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# 待优化
def noniid(dataset, traindataset, num_users):
    num_shards, num_imgs = 0, 0  # 设置默认值
    if dataset == 'mnist':
        num_shards, num_imgs = 200, 300
    elif dataset == 'cifar10':
        num_shards, num_imgs = 200, 250
    elif dataset == 'cifar100':
        num_shards, num_imgs = 200, 250
    elif dataset == 'stl10':
        num_shards, num_imgs = 20, 250
    elif dataset == 'svhn':
        num_shards, num_imgs = 24419, 3
    elif dataset == 'tiny':
        num_shards, num_imgs = 500, 200
    elif dataset == 'gtsrb':
        num_shards, num_imgs = 39209, 1
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = traindataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def iid_data(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users = {}
    all_idxs = np.random.permutation(len(dataset))
    for i in range(num_users):
        start_idx = i * num_items
        end_idx = start_idx + num_items
        dict_users[i] = set(all_idxs[start_idx:end_idx])
    return dict_users

def save_dataset_npz(dataset_test, file_path):
    images = []
    labels = []
    for image, label in dataset_test:
        images.append(np.array(image))
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    np.savez(file_path, images=images, labels=labels)
    
# 加载干净数据
def load_data_from_npz(dataset, i):
    npz_file = f'./data/{dataset}/client_{i}_clean_train.npz'
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    labels = loaded_data['labels']
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels).long()
    dataset = TensorDataset(data, labels)
    return dataset

# 加载评估数据
def load_eval_data_from_npz(dataset):
    npz_file = f'./data/{dataset}/clean_test.npz'
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    labels = loaded_data['labels']
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels).long()
    # print("Feature shape:", data.shape)
    dataset = TensorDataset(data, labels)
    return dataset

def get_dataset(dataset, iid=True, num_users=10):
    class DatasetSplit(Dataset):
        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = list(idxs)
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, item):
            image, label = self.dataset[self.idxs[item]]
            return image, label
    
    if dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=trans_mnist)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)


    elif dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dataset_train = datasets.FashionMNIST('./data/fmnist', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('./data/fmnist', train=False, download=True, transform=trans_fmnist)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
        dataset_test = datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'stl10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        dataset_train = datasets.STL10('./data/stl10', split='train', download=True, transform=transform)
        dataset_test = datasets.STL10('./data/stl10', split='test', download=True, transform=transform)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'tiny':
        transform = transforms.Compose([
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        id_dic = {}
        for i, line in enumerate(open('./data/tiny' + '/wnids.txt', 'r')):
            id_dic[line.replace('\n', '')] = i
        dataset_train = TrainTinyImageNet(root='./data/tiny',id=id_dic, transform=transform)
        dataset_test = ValTinyImageNet(root='./data/tiny', id=id_dic, transform=transform)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)

    elif dataset == 'gtsrb':
        data_folder = './data/gtsrb/DatasetFolder_png'
        transform = transforms.Compose([ToPILImage(),
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = DatasetFolder(root=os.path.join(data_folder, 'train'),
                                 loader=cv2.imread,
                                 extensions=('png',),
                                 transform=transform,
                                 target_transform=None,
                                 is_valid_file=None)

        dataset_test = DatasetFolder(root=os.path.join(data_folder, 'test'),
                                loader=cv2.imread,
                                extensions=('png',),
                                transform=transform,
                                target_transform=None,
                                is_valid_file=None)
        if iid:
            dict_users = iid_data(dataset_train, num_users)
        else:
            dict_users = noniid(dataset, dataset_train, num_users)
            

    dataset_test_path = f'./data/{dataset}/clean_test.npz' 
    save_dataset_npz(dataset_test, dataset_test_path)

    for i in range(num_users):
        train_data = DatasetSplit(dataset_train, dict_users[i])
        dataset_train_path = f'./data/{dataset}/client_{i}_clean_train.npz'
        save_dataset_npz(train_data, dataset_train_path)

    return dataset_train, dataset_test, dict_users
    
# 分离后门数据和剩余数据
def insert_backdoor(args, x_train_party, y_train_party, example_target, backdoor, i):
    if args.poi:
        percent_poison = args.ratio
        y_train_party_np = y_train_party
        all_indices = np.arange(len(x_train_party)) 
        
        #按一定比例进行后门
        remove_indices = all_indices[y_train_party_np == example_target]
        target_indices = list(set(all_indices) - set(remove_indices))
        num_poison = int(percent_poison * len(target_indices))
        selected_indices = np.random.choice(target_indices, num_poison, replace=False)
        remaining_indices = np.setdiff1d(all_indices, selected_indices)


        #未加后门的干净的数据
        clean_x = x_train_party[selected_indices]
        clean_y = y_train_party[selected_indices]
        # print("clean_data:", clean_x.shape, clean_y.shape)

        clean_data = TensorDataset(torch.Tensor(clean_x), torch.Tensor(clean_y).long())
        clean_data_path = f'./data/{args.dataset}/client_{i}_poi_origin_clean_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(clean_data, clean_data_path)

        x_np = x_train_party[selected_indices]
        x_np = np.transpose(x_np, (0, 2, 3, 1)) 
        y_np = np.array([example_target])
        poisoned_data, poisoned_labels = backdoor.poison(x_np, y=y_np, broadcast=True)
        poisoned_data = np.transpose(poisoned_data, (0, 3, 1, 2))
        print("target_data:", poisoned_data.shape, poisoned_labels.shape) # (537, 28, 28, 1) (537, 1)
        # 将后门数据摘出来
        poi_labels = np.squeeze(poisoned_labels)  # 尝试去除多余的维度
        poi_data = TensorDataset(torch.Tensor(poisoned_data), torch.Tensor(poi_labels).long())
        poi_data_path = f'./data/{args.dataset}/client_{i}_poi_target_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(poi_data, poi_data_path)

        # 将剩余数据摘出来
        remain_x = x_train_party[remaining_indices]
        remain_y = y_train_party[remaining_indices]
        print("remain_data:", remain_x.shape, remain_y.shape)
        remain_data = TensorDataset(torch.Tensor(remain_x), torch.Tensor(remain_y).long())
        remain_data_path = f'./data/{args.dataset}/client_{i}_poi_remain_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(remain_data, remain_data_path)

        # 后门数据和剩余数据合并
        poisoned_x_train = np.copy(x_train_party)
        poisoned_y_train = np.copy(y_train_party)
        for s, i in zip(selected_indices, range(len(selected_indices))):
            poisoned_x_train[s] = poisoned_data[i]
            poisoned_y_train[s] = int(poisoned_labels[i])
    else:
        # 不进行投毒
        percent_poison = args.ratio  # 0.1
        y_train_party_np = y_train_party.numpy()
        all_indices = np.arange(len(x_train_party))  

        # 和目标标签相同的索引
        selected_indices = all_indices[y_train_party_np == args.origin_label]
        # 获取剩余索引
        remaining_indices = np.setdiff1d(all_indices, selected_indices)

        # 将目标数据摘出来
        target_x = x_train_party[selected_indices]
        target_y = y_train_party[selected_indices]
        print("target_data:", target_x.shape, target_y.shape)
        target_data = TensorDataset(torch.Tensor(target_x), torch.Tensor(target_y).long())
        target_data_path = f'./data/{args.dataset}/client_{i}_clean_target_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(target_data, target_data_path)

        # 将剩余数据摘出来
        remain_x = x_train_party[remaining_indices]
        remain_y = y_train_party[remaining_indices]
        print("remain_data:", remain_x.shape, remain_y.shape)
        remain_data = TensorDataset(torch.Tensor(remain_x), torch.Tensor(remain_y).long())
        remain_data_path = f'./data/{args.dataset}/client_{i}_clean_remain_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(remain_data, remain_data_path)

        # 后门数据和剩余数据合并
        poisoned_x_train = np.copy(x_train_party)
        poisoned_y_train = np.copy(y_train_party)
    return poisoned_x_train, poisoned_y_train

def load_poidata(args, dataset, i):
    npz_file = f'./data/{dataset}/client_{i}_clean_train.npz'
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    labels = loaded_data['labels']

    # 创建后门模式
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    # 目标标签
    example_target = args.target_label
    
    
    
    if args.poi:
        # 插入后门模式
        poisoned_x_train, poisoned_y_train = insert_backdoor(args, data, labels, example_target, backdoor, i)
        poi_data_train_path = f'./data/{dataset}/client_{i}_poi_train_{args.target_label}_{args.ratio}.npz'
        poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train), torch.Tensor(poisoned_y_train).long())
        save_dataset_npz(poisoned_dataset_train, poi_data_train_path)
    else:
        # 不插入后门模式
        poisoned_x_train, poisoned_y_train = insert_backdoor(args, data, labels, example_target, backdoor, i)
        # print(poisoned_x_train.shape, poisoned_y_train.shape) #(6000, 1, 28, 28) (6000,)
        # print(type(poisoned_x_train), type(poisoned_y_train))  #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
        poi_data_train_path = f'./data/{dataset}/client_{i}_clean_train_{args.target_label}_{args.ratio}.npz'
        poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train),torch.Tensor(poisoned_y_train).long())
        save_dataset_npz(poisoned_dataset_train, poi_data_train_path)
# return poisoned_dataset_train

# 加载数据
def load_npz(npz_file):
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    labels = loaded_data['labels']
    dataset = TensorDataset(torch.Tensor(data), torch.Tensor(labels).long())
    return dataset

def add_trigger(args, npz_file):
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    data = torch.from_numpy(data)
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = args.target_label
    x_np = data.detach().cpu().numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))  # (537, 28, 28, 1)
    y_np = np.array([example_target])
    poisoned_data, poisoned_labels = backdoor.poison(x_np, y=y_np, broadcast=True)
    poisoned_data = np.transpose(poisoned_data, (0, 3, 1, 2))
    print("target_data:", poisoned_data.shape, poisoned_labels.shape)  # (537, 28, 28, 1) (537, 1)
    # 将后门数据摘出来
    poisoned_data=torch.Tensor(poisoned_data)
    return poisoned_data

def insert_backdoor_testdata(args, x_train_party, y_train_party, example_target, backdoor):
    if args.poi:
        # Insert backdoor
        percent_poison = args.ratio 
        y_train_party_np = y_train_party.numpy()
        all_indices = np.arange(len(x_train_party))
        # 将目标标签排除
        remove_indices = all_indices[y_train_party == example_target]
        # 选择剩余标签
        selected_indices = list(set(all_indices) - set(remove_indices))
        # 和目标标签相同的索引
        # selected_indices = all_indices[y_train_party_np == args.origin_label]
        x_np = x_train_party[selected_indices].detach().cpu().numpy()
        x_np = np.transpose(x_np, (0, 2, 3, 1))  # (537, 28, 28, 1)
        y_np = np.array([example_target])
        poisoned_data, poisoned_labels = backdoor.poison(x_np, y=y_np, broadcast=True)
        poisoned_data = np.transpose(poisoned_data, (0, 3, 1, 2))
        # 将后门数据摘出来
        poi_labels = np.squeeze(poisoned_labels)  # 尝试去除多余的维度
        poi_data = TensorDataset(torch.Tensor(poisoned_data), torch.Tensor(poi_labels).long())
        poi_data_path = f'./data/{args.dataset}/poi_test_{args.target_label}_{args.ratio}.npz'
        save_dataset_npz(poi_data, poi_data_path)

def load_testdata(args, dataset):
    npz_file = f'./data/{dataset}/clean_test.npz'
    loaded_data = np.load(npz_file)
    data = loaded_data['images']
    labels = loaded_data['labels']
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels).long()
    # 创建后门模式
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    # 目标标签
    example_target = args.target_label
    if args.poi:
        insert_backdoor_testdata(args, data, labels, example_target, backdoor)
