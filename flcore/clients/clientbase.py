import copy
import itertools
import random

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from torch import autograd
from torch.autograd import Variable
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import load_data_from_npz, load_npz, load_eval_data_from_npz
import torch.nn.init as init
from collections import defaultdict

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
# torch.backends.cudnn.deterministic = True

class Client(object):
    """
    Base class for clients in federated learning.
    """
    def __init__(self, args, id, train_samples, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model) #复制全局模型作为本地个性化模型的初始模型
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  
        self.save_folder_name = args.save_folder_name
        
        # 是否有必要
        self.num_classes = args.num_classes
        self.train_samples = train_samples #训练样本个数
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.ul_epochs = args.ul_epochs
        self.mode = args.mode
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.protos = None
        
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        #是否需要初始值
        # self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        # self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 是否通过参数传递
        self.loss = nn.CrossEntropyLoss()
        
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=0.9)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=1e-4)
        else:
            raise NotImplementedError
        
        
        self.learning_rate_decay = args.learning_rate_decay
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, 
                gamma=args.learning_rate_decay_gamma
            )
        else:
            self.learning_rate_scheduler = None
        

 
    def estimate_parameter_importance(self, trainloader, model):
        trainloader = self.load_poi_data()
        importance = dict()
        for k, p in self.model.named_parameters():
            importance[k] = torch.zeros_like(p).to(self.device)
        # 使用eval
        model.eval()
        
        model.to(self.device)
        num_examples = 0
        for image_batch, label_batch in trainloader:
            image_batch.to(self.device)
            label_batch.to(self.device)
            num_examples += image_batch.size(0)
            output = model(image_batch)
            loss = self.loss(output, label_batch)
            self.optimizer.zero_grad()
            loss.backward()
            for k, p in model.named_parameters():
                importance[k].add_(p.grad.abs())
        for k, v in importance.items():
            importance[k] = v.div(num_examples)
        return importance

    def teacher_model_init(self):
        num_models = 10  # 想要初始化的模型数量
        # 初始化10个ResNet模型并保存在列表中
        teacher_models = []
        for i in range(num_models):
            # 复制全局模型的权重
            new_model = copy.deepcopy(self.model)
            # 随机初始化新模型的权重
            for param in new_model.parameters():
                init.xavier_uniform_(param)
            teacher_models.append(new_model)
        # 打印模型列表中的每个模型结构
        # for idx, model in enumerate(teacher_models):
        #     print(f"Model {idx + 1}:")
        #     print(model)
        #
        # # 使用10个随机模型对每条数据预测标签然后取平均值作为新的标签
        # all_predictions = torch.zeros((self.batch_size, len(teacher_models)))
        # for idx, model in enumerate(teacher_models):
        #     model.eval()
        #     outputs = model(inputs)
        #     _, predicted = torch.max(outputs, 1)
        #     all_predictions[:, idx] = predicted
        # new_labels = torch.mean(all_predictions.float(), dim=1).long()  # 取预测标签的平均值作为新的标签

    def extract_features_from_dataloader(self, data_loader):
        #使用eval
        self.model.eval()
        
        self.model.to(self.device)
        all_features = []
        with torch.no_grad():
            for data in data_loader:
                inputs, _ = data  # 假设 DataLoader 返回的数据是 (inputs, labels) 格式
                inputs.to(self.device)  # 将数据移到设备上（根据你的设置）
                features = self.model.base(inputs)
                all_features.append(features.cpu())  # 将特征移动回 CPU（如果需要）
        all_features = torch.cat(all_features, dim=0)
        return all_features
    
    def wd_distance(self, feature1, feature2):
        distribution1 = feature1.detach().cpu().numpy()
        distribution2 = feature2.detach().cpu().numpy()
        wd_distance = wasserstein_distance(distribution1.flatten(), distribution2.flatten())
        return wd_distance
    
    # def compute_kl(pretrained_model, current_model, batch, device):
    #     """
    #     Compute *forward* KL as the normal utility loss.
    #
    #     Args:
    #         pretrained_model: reference model which is the pretrained (original) model.
    #         current_model: The current unlearning model.
    #         batch: A batch of normal data.
    #         device: GPU device.
    #
    #     Returns:
    #        The KL loss.
    #     """
    #     normal_outputs = current_model(
    #         batch["input_ids"].to(device),
    #         attention_mask=batch["attention_mask"].to(device),
    #         labels=batch["labels"].to(device),
    #     )
    #     with torch.no_grad():
    #         pretrained_outputs = pretrained_model(
    #             batch["input_ids"].to(device),
    #             attention_mask=batch["attention_mask"].to(device),
    #             labels=batch["labels"].to(device),
    #         )
    #     # P: pretrained model; Q: current model.
    #     prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    #     prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)
    #     loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
    #     return loss
    
    def load_target_test_data(self):
        # 更高效
        # def filter_dataset_by_label(dataset, label_to_filter):
        #     indices = [i for i, (_, label) in enumerate(dataset) if label == label_to_filter]
        #     # 使用索引创建一个新的子数据集
        #     filtered_dataset = torch.utils.data.Subset(dataset, indices)
        #     return filtered_dataset
        def filter_dataset_by_label(dataset, label_to_filter):
            filtered_data = []
            filtered_labels = []
            for data, label in dataset:
                if label == label_to_filter:
                    filtered_data.append(data)
                    filtered_labels.append(label)
            filtered_data = torch.stack(filtered_data)
            filtered_labels = torch.tensor(filtered_labels).long()
            filtered_dataset = TensorDataset(filtered_data, filtered_labels)
            return filtered_dataset
        dataset = load_eval_data_from_npz(self.dataset)
        # 选择标签为0的数据创建新的dataset
        new_dataset = filter_dataset_by_label(dataset, label_to_filter=self.args.origin_label)
        # new_dataset = filter_dataset_by_label(dataset, label_to_filter=5)
        return DataLoader(dataset=new_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

    def load_origin_clean_target_data(self):
        if self.args.poi:
            target_data = load_npz(f'./data/{self.dataset}/client_{self.id}_poi_origin_clean_{self.args.target_label}_{self.args.ratio}.npz')
        else:
            target_data = load_npz(f'./data/{self.dataset}/client_{self.id}_poi_origin_clean_{self.args.target_label}_{self.args.ratio}.npz')
        return DataLoader(dataset=target_data, batch_size=self.batch_size, drop_last=False, shuffle=False)

    # 加载干净的训练数据
    def load_train_data(self):
        train_data = load_data_from_npz(self.dataset, self.id)
        return DataLoader(dataset=train_data, batch_size=self.batch_size, drop_last=False, shuffle=True)
        
    # 加载有毒和干净整体的训练数据
    def load_poi_data(self):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{self.id}_poi_train_{self.args.target_label}_{self.args.ratio}.npz'
            target_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{self.id}_clean_train_{self.args.target_label}_{self.args.ratio}.npz'
            target_data = load_npz(npz_file)
        return DataLoader(dataset=target_data, batch_size=self.batch_size, drop_last=False, shuffle=True)
    # 加载剩余的干净数据
    def load_remain_data(self):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{self.id}_poi_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{self.id}_clean_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        return DataLoader(dataset=remain_data, batch_size=self.batch_size, drop_last=True, shuffle=True)
    # 加载有毒的目标数据
    def load_target_data(self):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{self.id}_poi_target_{self.args.target_label}_{self.args.ratio}.npz'
            target_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{self.id}_clean_target_{self.args.target_label}_{self.args.ratio}.npz'
            target_data = load_npz(npz_file)
        return DataLoader(dataset=target_data, batch_size= len(target_data), drop_last=True, shuffle=False)

    def fine_defense_adjust_learning_rate(self, optimizer, epoch, lr, dataset):
        if dataset=='cifar10':
            if epoch < 2:
                lr = 0.001
            elif epoch < 5:
                lr = 0.0001
            # elif epoch < 20:
            #     lr = 0.001
            else:
                lr = 0.001
        elif dataset=='gtsrb':
            if epoch < 2:
                lr = 0.01
            elif epoch < 10:
                lr = 0.001
            elif epoch < 20:
                lr = 0.0001
            else:
                lr = 0.0001
        else:
            raise Exception('Invalid dataset')
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def target_eval(self):
        if self.args.poi:
            target_data = load_npz(f'./data/{self.dataset}/client_{self.id}_poi_target_{self.args.target_label}_{self.args.ratio}.npz')
        else:
            target_data = load_npz(f'./data/{self.dataset}/client_{self.id}_clean_target_{self.args.target_label}_{self.args.ratio}.npz')
        
        testloader = DataLoader(dataset=target_data, batch_size=len(target_data), drop_last=False, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        poi_correct_predictions = 0
        poi_total_samples = 0
        predicted_labels_list = []  # Store predicted labels for each data point
        with torch.no_grad(): 
            for j, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted_labels_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                poi_correct_predictions += (torch.sum(torch.argmax(outputs, dim=1) == labels)).item()
                poi_total_samples += labels.size(0)
        poi_accuracy = poi_correct_predictions * 1.0 / poi_total_samples
        return poi_accuracy
    
    def remain_eval(self):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{self.id}_poi_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{self.id}_clean_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        
        testloader = DataLoader(dataset=remain_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        remain_acc = 0
        remain_num = 0
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                remain_acc += (torch.sum(torch.argmax(outputs, dim=1) == labels)).item()
                remain_num += labels.size(0)
        remain_accuracy = remain_acc * 1.0 / remain_num
        return remain_accuracy
        
    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())

    def add_trigger(self, data):
        data = data.detach().cpu().numpy()
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        example_target = self.args.target_label
        x_np = np.transpose(data, (0, 2, 3, 1))  # (537, 28, 28, 1)
        y_np = np.array([example_target])
        poisoned_data, poisoned_labels = backdoor.poison(x_np, y=y_np, broadcast=True)
        poisoned_data = np.transpose(poisoned_data, (0, 3, 1, 2))
        poisoned_data = torch.Tensor(poisoned_data)
        return poisoned_data

    # https://github.com/normal-crayon/FedCL
    # def ewc(self):
    #     trainloader = self.load_poi_data()
    #     tmp_fisher = {}
    #     self.model.eval()
    #     num_examples = 0
    #     for image_batch, label_batch in trainloader:
    #         image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)
    #         num_examples += image_batch.size(0)
    #         # Compute output and loss
    #         output = self.model(image_batch)
    #         loss = self.loss(output, label_batch)
    #         # Compute log-likelihoods
    #         # log_likelihoods = F.log_softmax(output, dim=1)[range(image_batch.size(0)), label_batch]
    #         # Compute gradients for log-likelihoods
    #         self.optimizer.zero_grad()
    #         loss.backward()  # Take the negative mean for minimization
    #         # (-log_likelihoods.mean()).backward()  # Take the negative mean for minimization
    #         gradients = {n: p.grad.detach() for n, p in self.model.named_parameters()}
    #         # Accumulate Fisher Information Matrix
    #         for name, param in self.model.named_parameters():
    #             tmp_fisher[name] += gradients[name].pow(2)
    #     for name in tmp_fisher:
    #         tmp_fisher[name] /= num_examples
    #     return tmp_fisher

    # FedCL进一步考虑了使用持续学习领域的弹性权重整合（EWC）的正则化局部损失函数中的参数重要性。然后，它们被转移到客户端，在那里进行惩罚步骤，以防止全局模型的重要参数在适应全局模型和客户的本地数据时被改变。这样做可以减轻本地和全局模型之间的权重差异，同时保留全局模型的知识以提高泛化能力。
    def ewc(self):
        trainloader = self.load_poi_data()
        tmp_weights = dict()
        for k, p in self.model.named_parameters():
            tmp_weights[k] = torch.zeros_like(p).to(self.device)
        self.model.eval()
        self.model.to(self.device)
        num_examples = 0
        for image_batch, label_batch in trainloader:
            image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)
            num_examples += image_batch.size(0)
            output = self.model(image_batch)
            loss = self.loss(output, label_batch)
            self.optimizer.zero_grad()
            loss.backward()
            for k, p in self.model.named_parameters():
                tmp_weights[k].add_(p.grad.detach() ** 2)
        for k, v in tmp_weights.items():
            tmp_weights[k] = torch.sum(v).div(num_examples)
        return tmp_weights

    def consolidate(self, data_loader):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        for x, y in data_loader:
            # x = x.view(batch_size, -1)
            x = Variable(x).to(self.device)
            y = Variable(y).type(torch.LongTensor).to(self.device)
            loglikelihoods.append(
                F.log_softmax(self.model(x), dim=1)[range(self.batch_size), y.data])  # self(x) the model's output
            if len(loglikelihoods) >= 2:
                break
        # estimate the fisher information of the parameters.
        # print('loglikelihoods',loglikelihoods1)
        loglikelihoods = torch.unbind(torch.cat(loglikelihoods))  # e.g. torch.unbind(torch.tensor([[1, 2, 3],[1, 2, 3]]) -> (tensor([1, 2, 3]), tensor([4, 5, 6]))
        # loglikelihoods = (tensor(1),tensor(2),tensor(3),tensor(1),tensor(2),tensor(3))
        # print('loglikelihoods',loglikelihoods)
        loglikelihood_grads = zip(*[autograd.grad(l, self.model.parameters(),retain_graph=(i < len(loglikelihoods))) for i, l in enumerate(loglikelihoods, 1)])
        print('loglikelihood_grads', loglikelihood_grads)
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        print('loglikelihood_grads', loglikelihood_grads)
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [n for n, p in self.model.named_parameters()]
        fisher = {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
        mean = {n: p.data for n, p in self.model.named_parameters()}
        return fisher, mean
    def freeze_norm_stats(self, net):
        try:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        except ValueError:  
            print("error with BatchNorm")
            return
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        self.model.to(self.device)
        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num, train_acc

    def train_poi_metrics(self):
        trainloader = self.load_poi_data()
        self.model.eval()
        self.model.to(self.device)
        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num, train_acc

    def train_remain_metrics(self):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{self.id}_poi_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{self.id}_clean_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        trainloader = DataLoader(dataset=remain_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
        self.model.eval()
        self.model.to(self.device)
        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num, train_acc

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def test_protos(self):
        def agg_func(protos):
            for [label, proto_list] in protos.items():
                if len(proto_list) > 1:
                    proto = 0 * proto_list[0].data
                    for i in proto_list:
                        proto += i.data
                    protos[label] = proto / len(proto_list)
                else:
                    protos[label] = proto_list[0]
            return protos
        # 加载测试数据
        dataset = load_eval_data_from_npz(self.dataset)
        testloader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 计算每个数据的表征
                rep = self.model.base(x)
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
        self.protos = agg_func(protos)
        return self.protos

