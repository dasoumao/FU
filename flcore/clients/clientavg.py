import copy
import random
import torch
import torch.nn as nn
import numpy as np
import time
from torch.autograd import Variable
from flcore.clients.clientbase import Client
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import ada_hessain
from utils.privacy import *
from utils.privacy import initialize_dp
from utils.privacy import get_dp_params
from itertools import cycle
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
# torch.backends.cudnn.deterministic = True

class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.tau = args.tau # 温度参数
        self.mu = args.mu  # 损失函数权重
        self.alpha = args.alpha  # 正则化项权重
        self.beta = args.beta  # 正则化项权重
        self.backup = None

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def ptrain(self):
        trainloader = self.load_poi_data()
        self.model.to(self.device)
        self.model.train()
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)
        start_time = time.time()
        for step in range(self.local_epochs):
            self.model.eval()
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        remain_acc=self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    # def ptrain(self):
    #     trainloader = self.load_remain_data()
    #     # poiloader = self.load_target_data()
    #     poiloader = self.load_poi_data()
    #     self.model.to(self.device)
    #     self.model.train()
    #     start_time = time.time()
    #     if self.privacy:
    #         model_origin = copy.deepcopy(self.model)
    #         self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
    #     for step in range(self.local_epochs): 
    #         for idx, (train_data, poi_data) in enumerate(zip(trainloader, cycle(poiloader)), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             if type(train_x) == type([]):
    #                 train_x[0] = train_x[0].to(self.device)
    #             else:
    #                 train_x = train_x.to(self.device)
    #             if type(poi_x) == type([]):
    #                 poi_x[0] = poi_x[0].to(self.device)
    #             else:
    #                 poi_x = poi_x.to(self.device)
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             train_output = self.model(train_x)
    #             poi_output = self.model(poi_x)
    #             train_loss = self.loss(train_output, train_y)
    #             poi_loss = self.loss(poi_output, poi_y.long())
    #             combined_loss = train_loss + poi_loss
    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()
    #     remain_acc=self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
    #     if self.learning_rate_decay:
    #         self.learning_rate_scheduler.step()
    #     self.train_time_cost['num_rounds'] += 1
    #     self.train_time_cost['total_cost'] += time.time() - start_time
    #     if self.privacy:
    #         eps, DELTA = get_dp_params(privacy_engine)
    #         print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
    #         for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
    #             param.data = param_dp.data.clone()
    #         self.model = model_origin
    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def remaintrain(self):
        trainloader = self.load_remain_data()
        self.model.to(self.device)
        self.model.train()
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        remain_acc = self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def adatrain(self):
        # 加载干净训练数据
        trainloader = self.load_remain_data()
        self.model.to(self.device)
        self.model.train()
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)
        self.optimizer_ada = ada_hessain.AdaHessian(self.model.parameters())
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_ada.zero_grad()
                loss.backward(create_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                self.optimizer_ada.step()
        remain_acc=self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def ewctrain(self):
        # 干净的训练数据
        trainloader = self.load_remain_data()
        # 有毒的训练数据
        poiloader = self.load_target_data()
        w_d = self.ewc()
        # print(w_d.keys())
        # # fisher, mean = self.consolidate(trainloader)
        # print(fisher)
        self.model.to(self.device)
        # 备份模型
        self.backup = copy.deepcopy(self.model)
        self.backup.load_state_dict(self.model.state_dict())
        self.model.train()  # 初始的全局模型训练模式
        self.model.to(self.device)
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)
        self.backup.to(self.device)
        self.backup.eval()  # 备份的全局模型评估模式
        start_time = time.time()
        for step in range(self.ul_epochs):
            correct = 0
            total = 0
            for idx, (train_data, poi_data) in enumerate(zip(trainloader, cycle(poiloader)), start=0):
                # 干净数据
                train_x, train_y = train_data
                # 有毒数据
                poi_x, poi_y = poi_data
                # 干净训练数据
                if type(train_x) == type([]):
                    train_x[0] = train_x[0].to(self.device)
                else:
                    train_x = train_x.to(self.device)
                # 有毒训练数据
                if type(poi_x) == type([]):
                    poi_x[0] = poi_x[0].to(self.device)
                else:
                    poi_x = poi_x.to(self.device)

                train_y = train_y.to(self.device)
                poi_y = poi_y.to(self.device)

                train_output = self.model(train_x)  # 干净训练数据的输出
                train_loss = self.loss(train_output, train_y)

                poi_output = self.model(poi_x) 
                poi_loss = - self.loss(poi_output, poi_y) 
                # losses = []
                # sum_ewc_loss = 0
                # for n, p in self.model.named_parameters():
                #     mean, omega = Variable(mean), Variable(fisher)
                #     losses.append((omega * (p - mean) ** 2).sum())
                # sum_ewc_loss += sum(losses)
                # ewc_loss = self.beta * sum_ewc_loss

                # 为了防止过度遗忘，引入正则化项
                proximal_term = 0.0
                for (name_model, param_model), (name_backup, param_backup) in zip(self.model.named_parameters(), self.backup.named_parameters()):
                    name_model = name_model.replace('_module.', '')  
                    proximal_term +=  torch.sum( w_d[name_model] * (param_backup - param_model) ** 2)
                reg_loss = (self.beta * 0.5) * proximal_term

                combined_loss = 0.5*train_loss + (1-0.5)*poi_loss + reg_loss  # 合并两个数据集的损失
                # combined_loss = poi_loss  # 合并两个数据集的损失

                self.optimizer.zero_grad()
                combined_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                self.optimizer.step()
                print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

        remain_acc=self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def backtrain(self):
        # 干净的训练数据
        trainloader = self.load_remain_data()
        # 有毒的训练数据
        poiloader = self.load_target_data()
        # 原始参数的备份
        origin_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        # 对模型进行深度复制
        model_for_importance = copy.deepcopy(self.model)
        # model_for_importance.load_state_dict(self.model.state_dict())
        model_for_importance1 = copy.deepcopy(self.model)
        # model_for_importance1.load_state_dict(self.model.state_dict())
        poi_num_samples= len(poiloader)
        remain_num_samples1 = len(trainloader)
        importance1 = self.estimate_parameter_importance(trainloader, model_for_importance1)
        # importance1 = self.estimate_parameter_importance(trainloader, model_for_importance1, remain_num_samples1)
        # importance = self.estimate_parameter_importance(model_for_importance, poiloader)
        # for keys in importance.keys():
        #     importance[keys] = (importance[keys] - importance[keys].min()) / (importance[keys].max() - importance[keys].min())
        #     importance[keys] = (1 - importance[keys])
        for keys in importance1.keys():
            importance1[keys] = (importance1[keys] - importance1[keys].min()) / (
                        importance1[keys].max() - importance1[keys].min())
            importance1[keys] = (1 - importance1[keys])

        self.model.to(self.device)
        self.model.train()  # 初始的全局模型训练模式
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)
        start_time = time.time()
        poiloader=cycle(poiloader)

        for step in range(self.ul_epochs):
            correct = 0
            total = 0
            for idx, (train_data, poi_data) in enumerate(zip(trainloader, poiloader), start=0):
                # 干净数据
                train_x, train_y = train_data
                # 有毒数据
                poi_x, poi_y = poi_data

                # 干净训练数据
                if type(train_x) == type([]):
                    train_x[0] = train_x[0].to(self.device)
                else:
                    train_x = train_x.to(self.device)
                # 有毒训练数据
                if type(poi_x) == type([]):
                    poi_x[0] = poi_x[0].to(self.device)
                else:
                    poi_x = poi_x.to(self.device)

                train_y = train_y.to(self.device)
                poi_y = poi_y.to(self.device)
                train_output = self.model(train_x)  # 干净训练数据的输出
                train_loss = self.loss(train_output, train_y)
                poi_output = self.model(poi_x)  # 干净训练数据的表征
                poi_loss = - self.loss(poi_output, poi_y)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                # 为了防止过度遗忘，引入正则化项
                reg_loss = 0.0
                for n, p in self.model.named_parameters():
                    # print((importance1[n]) * torch.abs(p - origin_params[n]))
                    reg_loss +=  torch.mean((importance1[n]) * torch.abs(p - origin_params[n]))/2
                combined_loss = 0.5*train_loss + (1-0.5)*poi_loss + reg_loss  # 合并两个数据集的损失
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()
                print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
            if self.privacy:
                eps, DELTA = get_dp_params(privacy_engine)
                print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
                for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                    param.data = param_dp.data.clone()
                self.model = model_origin
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        remain_acc = self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    def fliptrain(self):
        # 有毒的训练数据
        poiloader = self.load_target_data()
        self.model.to(self.device)
        self.model.train() # 初始的全局模型训练模式
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    poiloader, self.dp_sigma)
        start_time = time.time()
        for step in range(self.ul_epochs):
            correct = 0
            total = 0
            for idx, (poi_data) in enumerate((poiloader), start=0):
                # 有毒数据
                poi_x, poi_y = poi_data
                # 有毒训练数据
                if type(poi_x) == type([]):
                    poi_x[0] = poi_x[0].to(self.device)
                else:
                    poi_x = poi_x.to(self.device)
                # 生成不同于原始标签的新标签
                unique_labels = list(set(poi_y.tolist()))  # 获取原始标签的唯一值
                num_classes = 10 if self.dataset == "cifar10" or self.dataset == "fmnist" or self.dataset == "mnist" else 100
                choices = [i for i in range(num_classes) if i != unique_labels[0]]
                new_label = choices[random.randint(0, len(choices) - 1)]
                for i in range(len(poi_y)):
                    poi_y[i] = new_label
                poi_y = poi_y.to(self.device)
                poi_rep = self.model.base(poi_x)
                poi_output = self.model.head(poi_rep)
                poi_loss = self.loss(poi_output, poi_y)
                combined_loss = poi_loss
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
            if self.privacy:
                eps, DELTA = get_dp_params(privacy_engine)
                print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
                for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                    param.data = param_dp.data.clone()
                self.model = model_origin
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        remain_acc = self.remaineval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.targeteval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

# 原始版本
    # def ultrain(self):
    #     # if self.args.optimizer == 'sgd':
    #     #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
    #     #                                 momentum=0.9, weight_decay=1e-4)
    #     # elif self.args.optimizer == 'adam':
    #     #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
    #     #                                  weight_decay=1e-4)
    #     originloader= self.load_origin_clean_target_data()
    #     # 干净的训练数据
    #     trainloader = self.load_remain_data()
    #     # 有毒的训练数据
    #     poiloader = self.load_target_data()
    #     # 在干净的测试集中选择对应标签的数据
    #     targettestloader = self.load_target_test_data()
    #     # targettestloader = self.adv_train_data()
    #     self.model.to(self.device)
    #     self.model.train()  # 初始的全局模型训练模式
    #     correct = 0
    #     total = 0
    #     self.backup.to(self.device)
    #     self.backup.eval()  # 备份的全局模型评估模式
    #
    #     if self.privacy:
    #         model_origin = copy.deepcopy(self.model)
    #         self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,trainloader, self.dp_sigma)
    #     start_time = time.time()
    #
    #     for step in range(self.ul_epochs):
    #         cos = torch.nn.CosineSimilarity(dim=-1)  # 计算相似性
    #
    #         for idx, (train_data, poi_data, targettest, origin_target_clean) in enumerate(zip(trainloader, poiloader, targettestloader, originloader), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             t_x, t_y = targettest
    #             x, y = origin_target_clean
    #             if type(train_x) == type([]):
    #                 train_x[0] = train_x[0].to(self.device)
    #             else:
    #                 train_x = train_x.to(self.device)
    #             if type(poi_x) == type([]):
    #                 poi_x[0] = poi_x[0].to(self.device)
    #             else:
    #                 poi_x = poi_x.to(self.device)
    #             if type(t_x) == type([]):
    #                 t_x[0] = t_x[0].to(self.device)
    #             else:
    #                 t_x = t_x.to(self.device)
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             t_y = t_y.to(self.device)
    #             y = y.to(self.device)
    #
    #             train_output = self.model(train_x)
    #             train_loss = self.loss(train_output, train_y)
    #
    #             # negative_data= self.add_trigger(x).to(self.device)
    #             # # # 写法1
    #             # feature1 = self.model.base(negative_data) # 有毒数据的特征
    #             # feature2 = self.backup.base(x) # 同标签干净测试数据的表征：对主任务的影响较小，干净的测试数据中的相同标签的数据
    #             # poi_output = self.model.head(feature1)
    #             # posi = cos(feature1, feature2.detach())  # feature2.detach() 将 feature2 从计算图中分离出来，使其不再具有梯度信息。
    #             # logits = posi.reshape(-1, 1)  # 创建一个包含两个相似度值的 logits 张量，logits 的形状是 (batch_size, 2)
    #             # feature3 = self.backup.base(negative_data)  # 投毒数据的表征 效果最好
    #             # nega = cos(feature1, feature3.detach())
    #             # logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
    #             # print(logits)
    #             # logits /= self.tau
    #             # labels = torch.zeros(x.size(0)).cuda().long()  # 全零的一维张量, 这意味着所有的样本都被视为负样本
    #             # print(labels)
    #             # cmi_loss = self.loss(logits, labels)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
    #             # print(cmi_loss)
    #             # poi_loss = self.mu * cmi_loss
    #             # # 写法2
    #             # feature1 = self.model.base(negative_data)  # 有毒数据的特征
    #             # poi_output = self.model.head(feature1)
    #             # correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
    #             # total += poi_y.size(0)
    #             # feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             # feature3 = self.backup.base(negative_data).detach()  # 投毒数据的表征
    #             # cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #             #             torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #             #         F.cosine_similarity(feature1, feature3) / self.tau)))
    #             # poi_loss = self.mu * torch.mean(cmi_loss)
    #
    #             # 对比损失写法1
    #             # feature1 = self.model.base(poi_x)
    #             # poi_output = self.model.head(feature1)
    #             # feature2 = self.backup.base(t_x)
    #             # posi = cos(feature1, feature2.detach())
    #             # logits = posi.reshape(-1, 1)
    #             # feature3 = self.backup.base(poi_x)
    #             # nega = cos(feature1, feature3.detach())
    #             # logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
    #             # logits /= self.tau
    #             # labels = torch.zeros(poi_x.size(0)).cuda().long()
    #             # cmi_loss = self.loss(logits, labels)
    #             # poi_loss = self.mu * cmi_loss
    #
    #             # # 对比损失写法2
    #             feature1 = self.model.base(poi_x)  # 有毒数据的特征
    #             poi_output = self.model.head(feature1)
    #             feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             # feature3 = self.backup.base(train_x).detach()  # 投毒数据的表征
    #             feature3 = self.backup.base(poi_x).detach()  # 投毒数据的表征
    #             cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(F.cosine_similarity(feature1, feature3) / self.tau)))
    #             poi_loss = self.mu * torch.mean(cmi_loss)
    #
    #             correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
    #             total += poi_y.size(0)
    #
    #             # 为了防止过度遗忘，引入正则化项
    #             # proximal_term = 0.0
    #             # for w, w_t in zip(self.model.parameters(), self.backup.parameters()):
    #             #     proximal_term += torch.sum(torch.norm(w - w_t, p=2))
    #             # reg_loss = (self.alpha * 0.5) * proximal_term
    #
    #             # combined_loss = train_loss + poi_loss + reg_loss # 合并两个数据集的损失
    #             # combined_loss = poi_loss
    #             combined_loss = train_loss + poi_loss  # 合并两个数据集的损失
    #
    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()
    #
    #             accuracy = 1.0 * correct / total
    #             print("acc", accuracy)
    #             # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
    #             print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
    #             # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
    #
    #         if self.learning_rate_decay:
    #             self.learning_rate_scheduler.step()
    #
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time
    #         if self.privacy:
    #             eps, DELTA = get_dp_params(privacy_engine)
    #             print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
    #             for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
    #                 param.data = param_dp.data.clone()
    #             self.model = model_origin
    #             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    #
    #     remain_acc = self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    # def ultrain(self):
    #     # if self.args.optimizer == 'sgd':
    #     #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
    #     #                                 momentum=0.9, weight_decay=1e-4)
    #     # elif self.args.optimizer == 'adam':
    #     #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
    #     #                                  weight_decay=1e-4)
    #     trainloader = self.load_remain_data()
    #     # 有毒的训练数据
    #     poiloader = self.load_target_data()
    #     # 在干净的测试集中选择对应标签的数据
    #     targettestloader = self.load_target_test_data()
    #     self.model.to(self.device)
    #     self.model.train()  # 初始的全局模型训练模式
    #     correct = 0
    #     total = 0
    #     self.backup.to(self.device)
    #     self.backup.eval()  # 备份的全局模型评估模式
    #     for (param1, param2) in zip(self.model.parameters(), self.backup.parameters()):
    #         param1.requires_grad_(True)
    #         param2.requires_grad_(False)

    #     start_time = time.time()
    #     for step in range(self.ul_epochs):
    #         cos = torch.nn.CosineSimilarity(dim=-1)  # 计算相似性

    #         for idx, (train_data, poi_data, targettest) in enumerate(zip(trainloader, poiloader, targettestloader), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             t_x, t_y = targettest
    #             train_x = train_x.to(self.device)
    #             poi_x = poi_x.to(self.device)
    #             t_x = t_x.to(self.device)
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             # print(poi_y)
    #             t_y = t_y.to(self.device)

    #             train_output = self.model(train_x)
    #             train_loss = self.loss(train_output, train_y)

    #             negative_data= self.add_trigger(x).to(self.device)
    #             # # 写法1
    #             feature1 = self.model.base(negative_data) # 有毒数据的特征
    #             feature2 = self.backup.base(x) # 同标签干净测试数据的表征：对主任务的影响较小，干净的测试数据中的相同标签的数据
    #             poi_output = self.model.head(feature1)
    #             posi = cos(feature1, feature2.detach())  # feature2.detach() 将 feature2 从计算图中分离出来，使其不再具有梯度信息。
    #             logits = posi.reshape(-1, 1)  # 创建一个包含两个相似度值的 logits 张量，logits 的形状是 (batch_size, 2)
    #             feature3 = self.backup.base(negative_data)  # 投毒数据的表征 效果最好
    #             nega = cos(feature1, feature3.detach())
    #             logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
    #             print(logits)
    #             logits /= self.tau
    #             labels = torch.zeros(x.size(0)).cuda().long()  # 全零的一维张量, 这意味着所有的样本都被视为负样本
    #             print(labels)
    #             cmi_loss = self.loss(logits, labels)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
    #             print(cmi_loss)
    #             poi_loss = self.mu * cmi_loss
    #             # 写法2
    #             feature1 = self.model.base(negative_data)  # 有毒数据的特征
    #             poi_output = self.model.head(feature1)
    #             correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
    #             total += poi_y.size(0)
    #             feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             feature3 = self.backup.base(negative_data).detach()  # 投毒数据的表征
    #             cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #                         torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #                     F.cosine_similarity(feature1, feature3) / self.tau)))
    #             poi_loss = self.mu * torch.mean(cmi_loss)
    #             # 对比损失写法1
    #             # feature1 = self.model.base(poi_x)
    #             # poi_output = self.model.head(feature1)
    #             # feature2 = self.backup.base(t_x)
    #             # posi = cos(feature1, feature2.detach())
    #             # logits = posi.reshape(-1, 1)
    #             # feature3 = self.backup.base(poi_x)
    #             # nega = cos(feature1, feature3.detach())
    #             # logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
    #             # logits /= self.tau
    #             # labels = torch.zeros(poi_x.size(0)).cuda().long()
    #             # cmi_loss = self.loss(logits, labels)
    #             # poi_loss = self.mu * cmi_loss

    #             # # 对比损失写法2
    #             # feature1 = self.model.base(poi_x)  # 有毒数据的特征
    #             # # poi_labels = torch.argmax(poi_output, dim=1)
    #             # feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             # # cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2)))
    #             # # feature3 = self.backup.base(train_x).detach()  # 投毒数据的表征
    #             # feature3 = self.backup.base(poi_x).detach()  # 投毒数据的表征
    #             # cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #             #             torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #             #         F.cosine_similarity(feature1, feature3) / self.tau)))
    #             # poi_loss = self.mu * torch.mean(cmi_loss)
    #             # # print(poi_labels)


    #             # combined_loss = poi_loss
    #             combined_loss = train_loss + poi_loss  # 合并两个数据集的损失

    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()

    #             poi_output = self.model.head(feature1)
    #             correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
    #             total += poi_y.size(0)

    #             print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
    #             print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
    #             # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))

    #         if self.learning_rate_decay:
    #             self.learning_rate_scheduler.step()

    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time

    #     accuracy = 1.0 * correct / total
    #     print("acc", accuracy)
    #     remain_acc = self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    # def ultrain(self):
    #     originloader= self.load_origin_clean_target_data()
    #     # 干净的训练数据
    #     trainloader = self.load_remain_data()
    #     # 有毒的训练数据
    #     poiloader = self.load_target_data()
    #     # 在干净的测试集中选择对应标签的数据
    #     targettestloader = self.load_target_test_data()
    #     # targettestloader = self.adv_train_data()
    #     self.model.to(self.device)
    #     self.backup = copy.deepcopy(self.model)
    #     self.backup.to(self.device)
    #     for (param1, param2) in zip(self.model.parameters(), self.backup.parameters()):
    #         param1.requires_grad_(True)
    #         param2.requires_grad_(False) 
    #         # print(param1.data == param2.data)
    #     if self.privacy:
    #         model_origin = copy.deepcopy(self.model)
    #         self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,trainloader, self.dp_sigma)
    #     start_time = time.time()
    #     for step in range(self.ul_epochs):
    #         # cos = torch.nn.CosineSimilarity(dim=-1)  # 计算相似性
    #         self.model.train()  # 初始的全局模型训练模式
    #         self.backup.eval()  # 备份的全局模型评估模式
    #         # self.fine_defense_adjust_learning_rate(self.optimizer, step, self.learning_rate, self.dataset)
    #         # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9, weight_decay=1e-4)
    #         for idx, (train_data, poi_data, targettest, origin_target_clean) in enumerate(zip(trainloader, cycle(poiloader), cycle(targettestloader), cycle(originloader)), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             t_x, t_y = targettest
    #             x, y = origin_target_clean
    #             if type(train_x) == type([]):
    #                 train_x[0] = train_x[0].to(self.device)
    #             else:
    #                 train_x = train_x.to(self.device)
    #             if type(poi_x) == type([]):
    #                 poi_x[0] = poi_x[0].to(self.device)
    #             else:
    #                 poi_x = poi_x.to(self.device)
    #             if type(t_x) == type([]):
    #                 t_x[0] = t_x[0].to(self.device)
    #             else:
    #                 t_x = t_x.to(self.device)
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             t_y = t_y.to(self.device)
    #             y = y.to(self.device)
    
    #             train_output = self.model(train_x)
    #             train_loss = self.loss(train_output, train_y)
    
    #             # negative_data= self.add_trigger(x).to(self.device)
    #             # # # 写法1
    #             # feature1 = self.model.base(negative_data) # 有毒数据的特征
    #             # feature2 = self.backup.base(x) # 同标签干净测试数据的表征：对主任务的影响较小，干净的测试数据中的相同标签的数据
    #             # poi_output = self.model.head(feature1)
    #             # posi = cos(feature1, feature2.detach())  # feature2.detach() 将 feature2 从计算图中分离出来，使其不再具有梯度信息。
    #             # logits = posi.reshape(-1, 1)  # 创建一个包含两个相似度值的 logits 张量，logits 的形状是 (batch_size, 2)
    #             # feature3 = self.backup.base(negative_data)  # 投毒数据的表征 效果最好
    #             # nega = cos(feature1, feature3.detach())
    #             # logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
    #             # logits /= self.tau
    #             # labels = torch.zeros(x.size(0)).cuda().long()  # 全零的一维张量, 这意味着所有的样本都被视为负样本
    #             # cmi_loss = self.loss(logits, labels)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
    #             # poi_loss = self.mu * cmi_loss

    #             # # 写法2
    #             # feature1 = self.model.base(negative_data)  # 有毒数据的特征
    #             # feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             # feature3 = self.backup.base(negative_data).detach()  # 投毒数据的表征
    #             # cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #             #             torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #             #         F.cosine_similarity(feature1, feature3) / self.tau)))
    #             # poi_loss = self.mu * torch.mean(cmi_loss)
                
    #             feature1 = self.model.base(poi_x)  # 有毒数据的表征
    #             feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             feature3 = self.backup.base(poi_x).detach()  # 投毒数据的表征
    #             cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #                         torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #                     F.cosine_similarity(feature1, feature3) / self.tau)))
    #             poi_loss =  self.mu*torch.mean(cmi_loss)

    #             proximal_term = 0.0
    #             for w, w_t in zip(self.model.parameters(), self.backup.parameters()):
    #                 proximal_term += torch.sum(torch.norm(w - w_t, p=2))
    #             reg_loss = (self.alpha * 0.5) * proximal_term

    #             combined_loss =  (1 - 0.5) * train_loss + 0.5 * poi_loss + reg_loss
    #             # combined_loss =  poi_loss 

    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()
    #             # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
    #             print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
    #             # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
    #             target_acc = self.targeteval()
    #             print('Target_acc:{:.4f}'.format(target_acc))
    #         # target_acc = self.targeteval()
    #         # print('epoch{} Target_acc:{:.4f}'.format(step, target_acc))
    #         # remain_acc = self.remaineval()
    #         # print('eopch{} Remain_acc:{:.4f}'.format(step, remain_acc))
    #         if self.learning_rate_decay:
    #             self.learning_rate_scheduler.step()
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time

    #         if self.privacy:
    #             eps, DELTA = get_dp_params(privacy_engine)
    #             print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
    #             for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
    #                 param.data = param_dp.data.clone()
    #             self.model = model_origin
    #             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    #     remain_acc = self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
    
    # 没有用到类原型的代码
    # def ultrain(self):
    #     # 计算类原型
    #     self.protos= self.test_protos()
    #     trainloader = self.load_remain_data()
    #     poiloader = self.load_target_data()
    #     targettestloader = self.load_target_test_data()
    #     self.model.to(self.device)
    #     self.backup = copy.deepcopy(self.model)
    #     self.backup.eval()  # 备份的全局模型评估模式
    #     self.backup.to(self.device)
    #     self.model.train()  # 初始的全局模型训练模式
    #     for (param1, param2) in zip(self.model.parameters(), self.backup.parameters()):
    #         param1.requires_grad_(True)
    #         param2.requires_grad_(False) 
    #     if self.privacy:
    #         model_origin = copy.deepcopy(self.model)
    #         self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,trainloader, self.dp_sigma)
    #     start_time = time.time()
    #     for step in range(self.ul_epochs):
    #         # cos = torch.nn.CosineSimilarity(dim=-1)  # 计算相似性
    #         self.model.eval()
    #         # for param in (param for module in self.model.modules() if isinstance(module, (nn.BatchNorm2d, nn.Dropout)) for param in module.parameters()):
    #         #     param.requires_grad_(False)
    #         # for param in (param for module in self.model.modules() if isinstance(module, (nn.BatchNorm2d)) for param in module.parameters()):
    #         #     param.requires_grad_(False)
    #         # self.fine_defense_adjust_learning_rate(self.optimizer, step, self.learning_rate, self.dataset)
    #         for idx, (train_data, poi_data, targettest) in enumerate(zip(trainloader, cycle(poiloader), cycle(targettestloader)), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             t_x, t_y = targettest
    #             if type(train_x) == type([]):
    #                 train_x[0] = train_x[0].to(self.device)
    #             else:
    #                 train_x = train_x.to(self.device)
    #             if type(poi_x) == type([]):
    #                 poi_x[0] = poi_x[0].to(self.device)
    #             else:
    #                 poi_x = poi_x.to(self.device)
    #             if type(t_x) == type([]):
    #                 t_x[0] = t_x[0].to(self.device)
    #             else:
    #                 t_x = t_x.to(self.device)
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             t_y = t_y.to(self.device)

    #             train_output = self.model(train_x)
    #             train_loss = self.loss(train_output, train_y) * (train_x.shape[0] / (train_x.shape[0] + poi_x.shape[0]))

    #             feature1 = self.model.base(poi_x)  # 有毒数据的表征
    #             poi_output=self.model.head(feature1)
    #             feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             feature3 = self.backup.base(poi_x).detach()  # 投毒数据的表征
    #             cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #                         torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #                     F.cosine_similarity(feature1, feature3) / self.tau)))
    #             poi_loss = self.mu * torch.mean(cmi_loss) * (poi_x.shape[0] / (train_x.shape[0] + poi_x.shape[0]))
    #             print("111: ", torch.argmax(poi_output, dim=1))

    #             proximal_term = 0.0
    #             for w, w_t in zip(self.model.parameters(), self.backup.parameters()):
    #                 proximal_term +=  0.5 * self.alpha * torch.sum(torch.norm(w - w_t, p=2)) 
    #             reg_loss = proximal_term

    #             combined_loss = train_loss + poi_loss + reg_loss

    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()

    #             # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
    #             print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
    #             # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))

    #         # target_acc = self.targeteval()
    #         # print('epoch{} Target_acc:{:.4f}'.format(step, target_acc))
    #         # remain_acc = self.remaineval()
    #         # print('eopch{} Remain_acc:{:.4f}'.format(step, remain_acc))
    #         if self.learning_rate_decay:
    #             self.learning_rate_scheduler.step()
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time
    #         if self.privacy:
    #             eps, DELTA = get_dp_params(privacy_engine)
    #             print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
    #             for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
    #                 param.data = param_dp.data.clone()
    #             self.model = model_origin
    #             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    #     remain_acc = self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    # 20240112备份代码
    # def ultrain(self):
    #     # 计算类原型
    #     # self.protos= self.test_protos() # 键值对中值的形状为torch.Size([512])
    #     trainloader = self.load_remain_data()
    #     poiloader = self.load_target_data()
    #     targettestloader = self.load_target_test_data()
    #     self.model.to(self.device)
    #     self.backup = copy.deepcopy(self.model)
    #     self.backup.eval()  # 备份的全局模型评估模式
    #     self.backup.to(self.device)
    #     self.model.train()  # 初始的全局模型训练模式
    #     for (param1, param2) in zip(self.model.parameters(), self.backup.parameters()):
    #         param1.requires_grad_(True)
    #         param2.requires_grad_(False) 
    #     if self.privacy:
    #         model_origin = copy.deepcopy(self.model)
    #         self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,trainloader, self.dp_sigma)
    #     start_time = time.time()
    #     for step in range(self.ul_epochs):
    #         self.model.eval()
    #         for idx, (train_data, poi_data, targettest) in enumerate(zip(trainloader, cycle(poiloader), cycle(targettestloader)), start=0):
    #             train_x, train_y = train_data
    #             poi_x, poi_y = poi_data
    #             t_x, t_y = targettest
    #             if type(train_x) == type([]):
    #                 train_x[0] = train_x[0].to(self.device)
    #             else:
    #                 train_x = train_x.to(self.device)
    #             if type(poi_x) == type([]):
    #                 poi_x[0] = poi_x[0].to(self.device)
    #             else:
    #                 poi_x = poi_x.to(self.device)
    #             if type(t_x) == type([]):
    #                 t_x[0] = t_x[0].to(self.device)
    #             else:
    #                 t_x = t_x.to(self.device)
    #             train_y = train_y.to(self.device)
    #             poi_y = poi_y.to(self.device)
    #             t_y = t_y.to(self.device)

    #             train_output = self.model(train_x)
    #             train_loss = self.loss(train_output, train_y) * (train_x.shape[0] / (train_x.shape[0] + poi_x.shape[0]))

    #             feature1 = self.model.base(poi_x)  # 有毒数据的表征
    #             poi_output=self.model.head(feature1)
    #             proto_new = copy.deepcopy(feature1.detach()) # torch.Size([128, 512])
    #             # 遍历 poi_x 中的每个样本
    #             for i, yy in enumerate(poi_y):
    #                 y_c = yy.item()
    #                 # 类型1：获取标签不为原始标签的类原型
    #                 # proto_new[i, :] = self.protos[0].data
    #                 # 类型2：获取标签为原始标签的类原型
    #                 # proto_new[i, :] = self.protos[y_c].data
    #                 # 类型3：获取不同类原型中和feature1最近的原型
    #                 other_labels = [label for label in self.protos.keys() if label != y_c]
    #                 similarities = [F.cosine_similarity(self.protos[label].data, feature1[i, :].unsqueeze(0), dim=1) for label in other_labels]
    #                 closest_label_index = torch.argmax(torch.cat(similarities))
    #                 closest_label = other_labels[closest_label_index]
    #                 proto_new[i, :] = self.protos[closest_label].data
    #                 # 类型4：获取所有类原型中和feature1最近的原型
    #                 # similarities = [F.cosine_similarity(self.protos[label].data, feature1[i, :].unsqueeze(0), dim=1) for label in self.protos.keys()]
    #                 # closest_label_index = torch.argmax(torch.cat(similarities))
    #                 # closest_label = list(self.protos.keys())[closest_label_index]
    #                 # proto_new[i, :] = self.protos[closest_label].data
    #             feature2 = proto_new.detach()
    #             # feature2 = self.backup.base(t_x).detach()  # 测试数据的表征
    #             feature3 = self.backup.base(poi_x).detach()  # 投毒数据的表征
    #             cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
    #                         torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
    #                     F.cosine_similarity(feature1, feature3) / self.tau)))
    #             poi_loss = self.mu * torch.mean(cmi_loss) * (poi_x.shape[0] / (train_x.shape[0] + poi_x.shape[0]))
    #             # print("111: ", torch.argmax(poi_output, dim=1))

    #             proximal_term = 0.0
    #             for w, w_t in zip(self.model.parameters(), self.backup.parameters()):
    #                 proximal_term +=  0.5 * self.alpha * torch.sum(torch.norm(w - w_t, p=2)) 
    #             reg_loss = proximal_term

    #             combined_loss = train_loss + poi_loss + reg_loss

    #             self.optimizer.zero_grad()
    #             combined_loss.backward()
    #             self.optimizer.step()

    #             # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
    #             print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
    #             # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))

    #         if self.learning_rate_decay:
    #             self.learning_rate_scheduler.step()
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time
    #         if self.privacy:
    #             eps, DELTA = get_dp_params(privacy_engine)
    #             print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
    #             for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
    #                 param.data = param_dp.data.clone()
    #             self.model = model_origin
    #             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    #     remain_acc = self.remaineval()
    #     print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
    #     target_acc = self.targeteval()
    #     print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    def ultrain(self):
        originloader= self.load_origin_clean_target_data()
        trainloader = self.load_remain_data()
        self.model.to(self.device)
        self.backup = copy.deepcopy(self.model)
        self.backup.to(self.device)
        self.backup.eval() 
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,trainloader, self.dp_sigma)
        start_time = time.time()
        for step in range(self.ul_epochs):
            self.model.train()  # 初始的全局模型训练模式
            # self.model.apply(self.freeze_norm_stats)
            self.model.eval()
            correct = 0
            total = 0
            cos = torch.nn.CosineSimilarity(dim=-1)  # 计算相似性
            for idx, (train_data, poi_data) in enumerate(zip(trainloader, cycle(originloader))):
                train_x, train_y = train_data
                x, y = poi_data
                if type(train_x) == type([]):
                    train_x[0] = train_x[0].to(self.device)
                else:
                    train_x = train_x.to(self.device)
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                train_y = train_y.to(self.device)

                y = y.to(self.device)

                train_rep=self.model.base(train_x)
                train_output = self.model.head(train_rep)
                train_loss = self.loss(train_output, train_y) * (train_x.shape[0] / (train_x.shape[0] + x.shape[0]))

                negative_data= self.add_trigger(x).to(self.device)
                feature1 = self.model.base(negative_data)  # 有毒数据的特征
                # poi_output = self.model.head(feature1)
                # proto_new = copy.deepcopy(feature1.detach()) # torch.Size([128, 512])
                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     proto_new[i, :] = self.protos[y_c].data
                # feature2 = proto_new.detach()
                # feature3 = self.backup.base(negative_data).detach()  # 投毒数据的表征
                # cmi_loss = - torch.log(torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) / (
                #             torch.exp(F.cosine_similarity(feature1, feature2) / self.tau) + torch.exp(
                #         F.cosine_similarity(feature1, feature3) / self.tau)))
                # poi_loss = self.mu * torch.mean(cmi_loss) * (x.shape[0] / (train_x.shape[0] + x.shape[0]))
                # print("pre_y: ", torch.argmax(poi_output, dim=1))
                # print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                
                feature1 = self.model.base(negative_data)
                poi_output = self.model.head(feature1)
                proto_new = copy.deepcopy(feature1.detach()) # torch.Size([128, 512])
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    proto_new[i, :] = self.protos[y_c].data
                feature2 = proto_new.detach()
                posi = cos(feature1, feature2.detach())
                logits = posi.reshape(-1, 1)
                feature3 = self.backup.base(negative_data)
                nega = cos(feature1, feature3.detach())
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= self.tau
                labels = torch.zeros(negative_data.size(0)).cuda().long()
                cmi_loss = self.loss(logits, labels)
                poi_loss = self.mu * cmi_loss * (x.shape[0] / (train_x.shape[0] + x.shape[0]))
                # print("111: ", torch.argmax(poi_output, dim=1))
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))

                # # # 写法1
                # feature1 = self.model.base(negative_data) # 有毒数据的特征
                # feature2 = self.backup.base(x) # 同标签干净测试数据的表征：对主任务的影响较小，干净的测试数据中的相同标签的数据
                # poi_output = self.model.head(feature1)
                # posi = cos(feature1, feature2.detach())  # feature2.detach() 将 feature2 从计算图中分离出来，使其不再具有梯度信息。
                # logits = posi.reshape(-1, 1)  # 创建一个包含两个相似度值的 logits 张量，logits 的形状是 (batch_size, 2)
                # feature3 = self.backup.base(negative_data)  # 投毒数据的表征 效果最好
                # nega = cos(feature1, feature3.detach())
                # logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                # logits /= self.tau
                # labels = torch.zeros(x.size(0)).cuda().long()  # 全零的一维张量, 这意味着所有的样本都被视为负样本
                # cmi_loss = self.loss(logits, labels)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
                # poi_loss = self.mu * cmi_loss
                

                correct += (torch.sum(torch.argmax(poi_output, dim=1) == 5)).item()
                total += y.size(0)

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.backup.parameters()):
                    proximal_term +=  0.5 * self.alpha * torch.sum(torch.norm(w - w_t, p=2)) 
                reg_loss = proximal_term

                combined_loss = train_loss + poi_loss + reg_loss

                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
            if self.privacy:
                eps, DELTA = get_dp_params(privacy_engine)
                print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
                for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                    param.data = param_dp.data.clone()
                self.model = model_origin
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
        # remain_acc = self.remaineval()
        # print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        # target_acc = self.targeteval()
        # print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))