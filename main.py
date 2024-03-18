#!/usr/bin/env python
import copy
import random
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
from flcore.servers.serveravg import FedAvg
from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.data_utils import get_dataset
import torchvision.models as models
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# 可复现
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

warnings.simplefilter("ignore")


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        if args.mode != 7:
            print(f"\n============= Running time: {i}th =============")
        start = time.time()
        if model_str == "cnn": # non-convex
            if "mnist" == args.dataset:
                args.model = LeNet().to(args.device)
            elif "fmnist" == args.dataset:
                args.model = LeNet().to(args.device)
            elif "cifar10" == args.dataset:
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(args.device)
            elif "svhn" == args.dataset:
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(args.device)
            elif "stl10" == args.dataset:
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(args.device)
            elif "gtsrb" == args.dataset:
                args.model = mobilenet_v2(pretrained=False, num_classes=43).to(args.device) 
            elif "tiny" == args.dataset: 
                args.model = torchvision.models.resnet50(pretrained=False, num_classes=200).to(args.device)
            elif "cifar100" == args.dataset:
                args.model = torchvision.models.resnet34(pretrained=False, num_classes=100).to(args.device) 
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            
            #选择算法
            server = FedAvg(args, i)
        else:
            raise NotImplementedError

        # 有毒数据上训练
        if args.mode == 0:
            server.poi_train()
            print("rs_test_acc=", server.rs_test_acc)
            print("rs_target_acc=", server.rs_target_acc)
            print("rs_remain_acc=", server.rs_remain_acc)
            print("rs_train_loss=", server.rs_train_loss)
            print("rs_test_loss=", server.rs_test_loss)
        # 重训
        elif args.mode == 1:
            server.re_train()
            print("rs_test_acc=", server.rs_test_acc)
            print("rs_target_acc=", server.rs_target_acc)
            print("rs_remain_acc=", server.rs_remain_acc)
            print("rs_train_loss=", server.rs_train_loss)
            print("rs_test_loss=", server.rs_test_loss)
        # 在剩余数据上微调
        elif args.mode == 2:
            server.con_train()
        # # HFU ada 快速重训
        # elif args.mode == 3:
        #     server.hfu_train()
        # SGA-EWC 逆梯度商城，弹性权重巩固
        elif args.mode == 4:
            server.ewc_train()
        # Back-importance
        elif args.mode == 5:
            server.back_train()
        # 对后门数据标签翻转 flip
        elif args.mode == 6:
            server.flip_train()
        # 对比遗忘 cons
        elif args.mode == 7:
            server.ul_train()
        
        time_list.append(time.time()-start)

    # print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    # print("All done!")
    # reporter.report()

if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', "--mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-poi', "--poi", type=bool, default=True, help="Poisoned data or not")
    parser.add_argument('-ratio', '--ratio', type=float, default=0.1, help='Poi ratio of training data')
    parser.add_argument('-origin_label','--origin_label', type=int, default=0, help='class of origin label') # 原始标签的数据
    parser.add_argument('-target_label', '--target_label', type=int, default=5, help='class of target label') # 后门的目标标签
    parser.add_argument('-tc', "--target_clients_num", type=int, default= 1, help="number of target clients") # 目标客户端个数
    parser.add_argument('-nc', "--num_clients", type=int, default=10, help="Total number of clients")
    parser.add_argument('-iid', "--iid", type=bool, default=True, help="IID")
    
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ur', "--ul_rounds", type=int, default=1) # 遗忘训练的轮次
    parser.add_argument('-cr', "--con_rounds", type=int, default=1) # 继续训练的轮次
    parser.add_argument('-uls', "--ul_epochs", type=int, default=1, help="Multiple unlearning update steps in one local epoch.")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Multiple update steps in one local epoch.")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    
    parser.add_argument('-optimizer', '--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01, help="Local learning rate") #adam选择0.001，sgd选择0.01
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="Random ratio of clients per round")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=10)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    parser.add_argument('-dp', "--privacy", type=bool, default=False, help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    # 超参数
    parser.add_argument('-alpha', "--alpha", type=float, default=0.0001, help="Regularization weight")  # 文章中一般设置0.01
    parser.add_argument('-beta', "--beta", type=float, default=1.0, help="Ewc weight")
    parser.add_argument('-mu', "--mu", type=float, default=1.0,help="Contrastive weight")  # 对比损失的权重 1.0， 5.0，10.0 权重越大，遗忘的程度越大，同时不影响主任务,mu=1，遗忘的比较慢，效果差. mu=5，遗忘的比较快，效果好
    parser.add_argument('-tau', "--tau", type=float, default=0.5)  # 温度 0.05，0.1，0.5，1.0 值越大越区分不出来，最好的效果就是0.05，0.01干净数据也降低很多，但是最终也能回到正常的acc

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    if args.mode != 7:
        print("=" * 50)
        print("Global round: {}".format(args.global_rounds))
        print("Algorithm: {}".format(args.algorithm))
        print("Local batch size: {}".format(args.batch_size))
        print("Local steps: {}".format(args.local_epochs))
        print("Local learning rate: {}".format(args.local_learning_rate))
        print("Optimizer: {}".format(args.optimizer))
        print("Total number of clients: {}".format(args.num_clients))
        if args.learning_rate_decay:
            print("Local learning rate decay gamma: {}".format(args.learning_rate_decay_gamma))
        print("Clients join in each round: {}".format(args.join_ratio))
        print("Clients randomly join: {}".format(args.random_join_ratio))
        print("Running times: {}".format(args.times))
        print("Using device: {}".format(args.device))
        print("Using DP: {}".format(args.privacy))
        if args.privacy:
            print("Sigma for DP: {}".format(args.dp_sigma))
        if args.device == "cuda":
            print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("DLG attack: {}".format(args.dlg_eval))
        if args.dlg_eval:
            print("DLG attack round gap: {}".format(args.dlg_gap))
        print("Number of classes: {}".format(args.num_classes))
        print("Dataset: {}".format(args.dataset))
        print("Model: {}".format(args.model))
        print("Mode: {}".format(args.mode))
        print("Target_label: {}".format(args.target_label))
        print("Origin_label: {}".format(args.origin_label))
        print("Poi: {}".format(args.poi))
        print("IID: {}".format(args.iid))
        print("Tau: {}".format(args.tau))
        print("Mu: {}".format(args.mu))
        print("=" * 50)

    run(args)
    