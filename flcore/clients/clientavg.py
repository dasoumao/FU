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
from itertools import cycle

from ..optimizers import get_optimizer

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

    
    #
    #下面的训练流程需要考察
    def train(self):
        trainloader = self.load_train_data()
        self.client_model.to(self.device)
        self.client_model.train()
        
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # 是否必要
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.client_model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if self.learning_rate_decay:
                learning_rate_scheduler.step()
            
        # self.train_time_cost['num_rounds'] += 1
        # self.train_time_cost['total_cost'] += time.time() - start_time
        
    def ptrain(self):
        # poi data 生成方式
        trainloader = self.load_poi_data()
        self.client_model.to(self.device)
        self.client_model.train()
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
        start_time = time.time()
        for step in range(self.local_epochs):
            
            self.client_model.eval()
            
            for i, (x, y) in enumerate(trainloader):
                # 是否必要
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.client_model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if self.learning_rate_decay:
                learning_rate_scheduler.step()
                
        remain_acc=self.remain_eval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.target_eval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))
        
        # self.train_time_cost['num_rounds'] += 1
        # self.train_time_cost['total_cost'] += time.time() - start_time

    def remaintrain(self):
        trainloader = self.load_remain_data()
        self.client_model.to(self.device)
        self.client_model.train()
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # 是否必要
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.client_model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if self.learning_rate_decay:
                learning_rate_scheduler.step()
        
        
        
        # self.train_time_cost['num_rounds'] += 1
        # self.train_time_cost['total_cost'] += time.time() - start_time

    def ewctrain(self):
        # 干净的训练数据
        trainloader = self.load_remain_data()
        # 有毒的训练数据
        poiloader = self.load_target_data()
        
        w_d = self.ewc()
        
        self.client_model.to(self.device)
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        # 备份模型
        self.backup = copy.deepcopy(self.client_model)
        self.backup.load_state_dict(self.client_model.state_dict())
        
        self.client_model.train()  # 初始的全局模型训练模式
        self.client_model.to(self.device)
        
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

                train_output = self.client_model(train_x)  # 干净训练数据的输出
                train_loss = self.loss(train_output, train_y)

                poi_output = self.client_model(poi_x) 
                poi_loss = - self.loss(poi_output, poi_y) 
                
                # losses = []
                # sum_ewc_loss = 0
                # for n, p in self.client_model.named_parameters():
                #     mean, omega = Variable(mean), Variable(fisher)
                #     losses.append((omega * (p - mean) ** 2).sum())
                # sum_ewc_loss += sum(losses)
                # ewc_loss = self.beta * sum_ewc_loss

                # 为了防止过度遗忘，引入正则化项
                proximal_term = 0.0
                for (name_model, param_model), (name_backup, param_backup) in zip(self.client_model.named_parameters(), self.backup.named_parameters()):
                    name_model = name_model.replace('_module.', '')  
                    proximal_term +=  torch.sum( w_d[name_model] * (param_backup - param_model) ** 2)
                reg_loss = (self.beta * 0.5) * proximal_term

                combined_loss = 0.5*train_loss + (1-0.5)*poi_loss + reg_loss  # 合并两个数据集的损失
                # combined_loss = poi_loss  # 合并两个数据集的损失

                optimizer.zero_grad()
                combined_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer.step()
                print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break

            if self.learning_rate_decay:
                learning_rate_scheduler.step()
            
            # self.train_time_cost['num_rounds'] += 1
            # self.train_time_cost['total_cost'] += time.time() - start_time

        remain_acc=self.remain_eval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.target_eval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    def backtrain(self):
        # 干净的训练数据
        trainloader = self.load_remain_data()
        # 有毒的训练数据
        poiloader = self.load_target_data()
        # 原始参数的备份
        origin_params = {n: p.clone().detach() for n, p in self.client_model.named_parameters()}
        # 对模型进行深度复制
        model_for_importance = copy.deepcopy(self.client_model)
        # model_for_importance.load_state_dict(self.client_model.state_dict())
        model_for_importance1 = copy.deepcopy(self.client_model)
        # model_for_importance1.load_state_dict(self.client_model.state_dict())
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

        self.client_model.to(self.device)
        self.client_model.train()  # 初始的全局模型训练模式
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
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
                train_output = self.client_model(train_x)  # 干净训练数据的输出
                train_loss = self.loss(train_output, train_y)
                poi_output = self.client_model(poi_x)  # 干净训练数据的表征
                poi_loss = - self.loss(poi_output, poi_y)  # 计算交叉熵损失 cmi_loss，该损失用于衡量模型的预测与目标标签的差异
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                # 为了防止过度遗忘，引入正则化项
                reg_loss = 0.0
                for n, p in self.client_model.named_parameters():
                    # print((importance1[n]) * torch.abs(p - origin_params[n]))
                    reg_loss +=  torch.mean((importance1[n]) * torch.abs(p - origin_params[n]))/2
                combined_loss = 0.5*train_loss + (1-0.5)*poi_loss + reg_loss  # 合并两个数据集的损失
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break

            if self.learning_rate_decay:
                learning_rate_scheduler.step()
            # self.train_time_cost['num_rounds'] += 1
            # self.train_time_cost['total_cost'] += time.time() - start_time

        remain_acc = self.remain_eval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.target_eval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    def fliptrain(self):
        # 有毒的训练数据
        poiloader = self.load_target_data()
        self.client_model.to(self.device)
        self.client_model.train() # 初始的全局模型训练模式
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
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
                poi_rep = self.client_model.base(poi_x)
                poi_output = self.client_model.head(poi_rep)
                poi_loss = self.loss(poi_output, poi_y)
                combined_loss = poi_loss
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                print('Idx: %d POI_Loss: %f' % (idx, poi_loss.item()))
                correct += (torch.sum(torch.argmax(poi_output, dim=1) == poi_y)).item()
                total += poi_y.size(0)
                acc = 1.0 * correct / total
                if acc < 0.1:
                    break
            if self.learning_rate_decay:
                learning_rate_scheduler.step()
            # self.train_time_cost['num_rounds'] += 1
            # self.train_time_cost['total_cost'] += time.time() - start_time
            

        remain_acc = self.remain_eval()
        print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        target_acc = self.target_eval()
        print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))

    def ultrain(self):
        originloader= self.load_origin_clean_target_data()
        trainloader = self.load_remain_data()
        self.client_model.to(self.device)
        self.backup = copy.deepcopy(self.client_model)
        self.backup.to(self.device)
        self.backup.eval() 
        optimizer = get_optimizer(self.optimizer, self.client_model.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.learning_rate_decay_gamma
            )
        else:
            learning_rate_scheduler = None
        
        start_time = time.time()
        for step in range(self.ul_epochs):
            self.client_model.train()  # 初始的全局模型训练模式
            # self.client_model.apply(self.freeze_norm_stats)
            self.client_model.eval()
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

                train_rep=self.client_model.base(train_x)
                train_output = self.client_model.head(train_rep)
                train_loss = self.loss(train_output, train_y) * (train_x.shape[0] / (train_x.shape[0] + x.shape[0]))

                negative_data= self.add_trigger(x).to(self.device)
                feature1 = self.client_model.base(negative_data)  # 有毒数据的特征
                # poi_output = self.client_model.head(feature1)
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
                
                feature1 = self.client_model.base(negative_data)
                poi_output = self.client_model.head(feature1)
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
                # feature1 = self.client_model.base(negative_data) # 有毒数据的特征
                # feature2 = self.backup.base(x) # 同标签干净测试数据的表征：对主任务的影响较小，干净的测试数据中的相同标签的数据
                # poi_output = self.client_model.head(feature1)
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
                for w, w_t in zip(self.client_model.parameters(), self.backup.parameters()):
                    proximal_term +=  0.5 * self.alpha * torch.sum(torch.norm(w - w_t, p=2)) 
                reg_loss = proximal_term

                combined_loss = train_loss + poi_loss + reg_loss

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                # print('Idx: %d Train_Loss: %f' % (idx, train_loss.item()))
                # print('Idx: %d Reg_Loss: %f' % (idx, reg_loss.item()))

            if self.learning_rate_decay:
                learning_rate_scheduler.step()
            # self.train_time_cost['num_rounds'] += 1
            # self.train_time_cost['total_cost'] += time.time() - start_time
    
        # remain_acc = self.remaineval()
        # print('C{} Remain_acc:{:.4f}'.format(self.id, remain_acc))
        # target_acc = self.targeteval()
        # print('C{} Target_acc:{:.4f}'.format(self.id, target_acc))