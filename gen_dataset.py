#!/usr/bin/env python
import random
import torch
import argparse
import time
import warnings
import numpy as np
import logging
from utils.data_utils import get_dataset, load_poidata
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")

if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-poi', "--poi", type=bool, default=True, help="Poisoned data or not")
    parser.add_argument('-ratio', '--ratio', type=float, default=0.1, help='Poi ratio of training data')
    parser.add_argument('-origin_label','--origin_label', type=int, default=0, help='class of origin label') # 原始标签的数据
    parser.add_argument('-target_label', '--target_label', type=int, default=5, help='class of target label') # 后门的目标标签
    parser.add_argument('-iid', "--iid", type=bool, default=True, help="IID")
    parser.add_argument('-nc', "--num_clients", type=int, default=10, help="Total number of clients")

    args = parser.parse_args()

    print("=" * 50)

    print("Total number of clients: {}".format(args.num_clients))
    print("Dataset: {}".format(args.dataset))
    print("Target_label: {}".format(args.target_label))
    print("Origin_label: {}".format(args.origin_label))
    print("Poison: {}".format(args.poi))
    print("Poison ratio: {}".format(args.ratio))
    print("IID: {}".format(args.iid))
    print("=" * 50)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    
    # 获取数据
    train_datasets, eval_datasets, dict_users = get_dataset(dataset=args.dataset, iid=args.iid, num_users=args.num_clients)
    
    # 目标客户端ID
    target_id = [0]
    # 生成后门数据
    for id in target_id:
        load_poidata(args, args.dataset, id)

    print("Gen Target Data End")
