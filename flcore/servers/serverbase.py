import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import read_client_data, load_data_from_npz, load_eval_data_from_npz, load_npz

# from utils.dlg import DLG

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

#可复现
# torch.backends.cudnn.deterministic = True

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.poi = args.poi  # 是否注入毒数据
        self.mode = args.mode  # 当前训练模式
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.con_rounds = args.con_rounds
        self.ul_rounds = args.ul_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)  # 全局模型
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.target_num = args.target_clients_num  # 目标客户端数量

        self.target_id = []  # 目标客户端id
        self.clients = []

        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_target_acc = []
        self.rs_remain_acc = []
        self.rs_test_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        # self.eval_loader = torch.utils.data.DataLoader(eval_datasets, batch_size=self.batch_size, shuffle=True)

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data = load_data_from_npz(self.dataset, i)
            client = clientObj(self.args, id=i, train_samples=len(train_data))
            self.clients.append(client)

    # 待改进
    def select_clients(self): # 参与训练的客户端
        self.selected_clients = self.clients
        return self.selected_clients

    def set_target_clients(self, target_id=[0]): # 设置投毒客户端
        self.target_id = target_id
        if self.args.mode != 7:
            print("Target_client_id:", self.target_id)
        return self.target_id

    def select_target_clients(self): #选择投毒客户端
        selected_clients = [self.clients[idx] for idx in self.target_id]
        return selected_clients

    #待改进
    def send_models(self): 
        assert (len(self.clients) > 0), "Set client first!"
        for client in self.selected_clients:
            # for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    #待改进
    def receive_models(self):
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        '''
        聚合模型参数，可根据需要修改聚合方式
        '''
        assert (len(self.uploaded_models) > 0)
        if self.mode == 4 or self.mode == 5 or self.mode == 6 or self.mode == 7:
            self.global_model = copy.deepcopy(self.uploaded_models[0])
        else:
            for param in self.global_model.parameters():
                param.data.zero_()
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
            
    # 文件管理

    #待改进，是否需要这么多数据
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, str(self.mode) + "_" + self.algorithm + "_server" + "_" + 
                                  str(self.args.target_label) + "_" + str(self.args.poi) + ".pt")
        torch.save(self.global_model, model_path)

    def save_init_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "init" + "_" + self.algorithm + "_server" + "_" + 
                                  str(self.args.target_label) + "_" + str(self.args.poi) + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, str(0) + "_" + self.algorithm + "_server" + "_" + 
                                  str(self.args.target_label) + "_" + str(self.args.poi) + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path).to(self.device)

    def load_init_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, "init" + "_" + self.algorithm + "_server" + "_" + 
                                  str(self.args.target_label) + "_" + str(self.args.poi) + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path).to(self.device)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    #待改进，result路径
    def save_results(self):
        assert(len(self.rs_test_acc)>0)
        
        path = self.dataset + "_" + self.algorithm + "_" + self.goal + "_" + str(self.times)
        result_path = "../results/"
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        file_path = result_path + "{}.h5".format(path)
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_remain_acc', data=self.rs_remain_acc)
            
    # 与dataset类一起更改
    def load_test_data(self):
        eval_data = load_eval_data_from_npz(self.dataset)
        return DataLoader(dataset=eval_data, batch_size=self.batch_size, drop_last=False, shuffle=False)

    def load_train_data(self, id):
        train_data = load_data_from_npz(self.dataset, id)
        return DataLoader(dataset=train_data, batch_size=self.batch_size, drop_last=False, shuffle=False)

    def load_remain_data(self, id):
        if self.args.poi:
            npz_file = f'./data/{self.dataset}/client_{id}_poi_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        else:
            npz_file = f'./data/{self.dataset}/client_{id}_clean_remain_{self.args.target_label}_{self.args.ratio}.npz'
            remain_data = load_npz(npz_file)
        return DataLoader(dataset=remain_data, batch_size=self.batch_size, drop_last=False, shuffle=False)

    # targetid影响
    def load_target_data(self):
        # for idx in self.target_id:
        if self.args.poi:
            target_data = load_npz(f'./data/{self.dataset}/client_{0}_poi_target_{self.args.target_label}_{self.args.ratio}.npz')
        else:
            target_data = load_npz(f'./data/{self.dataset}/client_{0}_clean_target_{self.args.target_label}_{self.args.ratio}.npz')
        return DataLoader(dataset=target_data, batch_size=len(target_data), drop_last=False, shuffle=False)

    # 通用保存
    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    
    # 获取参数， 待优化
    def basic_train_metrics(self):
        """
        基本参数
        """
        acc_num = []
        num_samples = []
        losses = []
        for c in self.selected_clients:
            if c.id in self.target_id:
                cl, ns, ac = c.train_poi_metrics()
                # cl, ns, ac = c.train_remain_metrics()
            else:
                cl, ns, ac = c.train_metrics()
            num_samples.append(ns)
            acc_num.append(ac)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses, acc_num

    def metrics(self, trainloader):
        '''
        获取准确率
        '''
        self.global_model.eval()
        self.global_model.to(self.device)
        train_acc = 0
        train_num = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
        return train_num, train_acc

    def remain_metrics(self):
        '''
        使用剩余数据获取数据集
        '''
        acc_num = []
        num_samples = []
        for c in self.selected_clients:
            if c.id in self.target_id:
                dataloader = self.load_remain_data(c.id)
                ns, ac = self.metrics(dataloader)
            else:
                dataloader = self.load_train_data(c.id)
                ns, ac = self.metrics(dataloader)
            num_samples.append(ns)
            acc_num.append(ac)
        ids = [c.id for c in self.clients]
        return ids, num_samples, acc_num

    def model_eval(self, dataloader, loss_fn):
        loss_function = loss_fn.to(self.device)
        self.global_model.to(self.device)
        self.global_model.eval()
        total_loss = 0.0
        test_acc = 0
        test_num = 0
        # predicted_labels = []  # 用于存储每个数据的预测标签
        with torch.no_grad():
            for batch_id, batch in enumerate(dataloader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.global_model(data)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == target)).item()
                test_num += target.shape[0]
                loss = loss_function(output, target)
                total_loss += loss.item() * target.shape[0]
        test_acc = float(test_acc) / test_num
        test_loss = total_loss / test_num
        return test_acc, test_loss

    def evaluate(self, loss_fn=nn.CrossEntropyLoss(), acc_list=None, loss_list=None):
        # poi_test_loader= self.load_poi_test()
        # asr, _ = self.model_eval(poi_test_loader)
        # print("BAtest asr: {:.4f}".format(asr))

        # 目标训练集
        target_loader = self.load_target_data()
        targetacc, _ = self.model_eval(target_loader, loss_fn)
        print("Target acc: {:.4f}".format(targetacc))

        # 测试集
        test_loader = self.load_test_data()
        testacc, test_loss = self.model_eval(test_loader, loss_fn)
        print("Test acc: {:.4f}".format(testacc))

        train = self.remain_metrics()
        remain_train_acc = sum(train[2]) * 1.0 / sum(train[1])
        print("Remain acc: {:.4f}".format(remain_train_acc))
        # accs = [a / n for a, n in zip(train[2], train[1])]
        # print(accs)

        stats_train = self.train_metrics()
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        train_acc = sum(stats_train[3]) * 1.0 / sum(stats_train[1])

        if acc_list == None:
            self.rs_test_acc.append(testacc)
            self.rs_target_acc.append(targetacc)
            self.rs_remain_acc.append(remain_train_acc)
        else:
            acc_list.append(testacc)

        if loss_list == None:
            self.rs_train_loss.append(train_loss)
            self.rs_test_loss.append(test_loss)
        else:
            loss_list.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        # print("Averaged Test Acc: {:.4f}".format(testacc))
        # print("Averaged Remain Train Acc: {:.4f}".format(remain_train_acc))