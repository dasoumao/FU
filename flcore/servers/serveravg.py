import random
import time
import numpy as np
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import load_poidata
import logging

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# 可复现
# torch.backends.cudnn.deterministic = True

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        #待改进
        # 创建客户端
        self.set_clients(clientAVG)
        # 设置投毒客户端id
        self.set_target_clients()
        
        if self.args.mode != 7:
            print("Create server and clients.")
            if self.args.poi:
                print("Use poi data.")
            else:
                print("Use clean data.")
        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")

        self.Budget = []

    def poi_train(self):
        # 保存随机生成的初始化模型，是否必要
        self.save_init_global_model()
        
        begin = time.time()
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 选择参与训练的模型
            self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Train Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                if client.id in self.target_id:
                    client.ptrain() # 目标客户端
                else:
                    client.train() # 剩余客户端
                    
            self.receive_models()
            
            #待优化
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
                
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        end = time.time()
        print('-' * 25, 'overall time cost', '-' * 25, end - begin)

            # 未使用
            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

                
        print("\nEvaluate global model.")
        self.evaluate()

        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def re_train(self):
        '''
        重新训练
        '''
        begin = time.time()
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Retrain Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                if client.id in self.target_id:
                    client.remaintrain() # 目标客户端
                else:
                    client.train() # 剩余客户端

            self.receive_models()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
                
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        end = time.time()
        print('-' * 25, 'overall time cost', '-' * 25, end - begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        print("\nEvaluate global model.")
        self.evaluate()

        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def con_train(self):
        '''
        干净数据上继续训练
        '''
        print("Load origin global model")
        self.load_model()
        
        # 在干净数据上继续微调
        for i in range(self.con_rounds):
            s_t = time.time()
            # self.selected_clients = self.select_target_clients()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Con Round number: {i+1}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            begin = time.time()
            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                if client.id in self.target_id:
                    client.remaintrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
            
            self.receive_models()

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
                
            self.aggregate_parameters()

            print("-------------After-------------")
            self.evaluate()
        
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
        end = time.time()
        print('-'*25, 'overall time cost', '-'*25, end-begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
        print("\nEvaluate global model.")
        self.evaluate()
        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nWorst target accuracy.")
        print(min(self.rs_target_acc))
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))
        self.save_results()
        # 保存最终全局模型
        self.save_global_model()

    def hfu_train(self):

        print("Load origin global model")
        self.load_init_model()

        begin = time.time()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------HFU Round number: {i + 1}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                if client.id in self.target_id:
                    client.adatrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
            
            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            # print("-------------After-------------")
            # self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
        end = time.time()
        print('-' * 25, 'overall time cost', '-' * 25, end - begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

            # # 保存每一轮的全局遗忘模型
            # self.save_ulandcon_model(i)
        print("\nEvaluate global model.")
        self.evaluate()
        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # 保存最终全局模型
        self.save_global_model()

    def ewc_train(self):

        print("\nLoad origin global model")
        self.load_model()

        for i in range(self.ul_rounds):
            s_t = time.time()
            self.selected_clients = self.select_target_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------EWC Round number: {i + 1}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                begin = time.time()
                if client.id in self.target_id:
                    client.ewctrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
                

            # 加权聚合,才是的数据量仍然为加权的数据量
            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            
            print("-------------After-------------")
            self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
        end = time.time()
        print('-' * 25, 'overall time cost', '-' * 25, end - begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        print("\nEvaluate global model.")
        self.evaluate()
        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # 保存最终全局模型
        self.save_global_model()

    def back_train(self):

        print("\nLoad origin global model")
        self.load_model()

        for i in range(self.ul_rounds):
            s_t = time.time()
            self.selected_clients = self.select_target_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Backdoor Round number: {i + 1}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                begin = time.time()
                if client.id in self.target_id:
                    client.backtrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
                

            # 加权聚合,才是的数据量仍然为加权的数据量
            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            
            print("-------------After-------------")
            self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'total time', '-' * 25, self.Budget[-1])
        end = time.time()
        print('-' * 25, 'time cost', '-' * 25, end - begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

            # # 保存每一轮的全局遗忘模型
            # self.save_ulandcon_model(i)
        print("\nEvaluate global model.")
        self.evaluate()
        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # 保存最终全局模型
        self.save_global_model()

    def flip_train(self):

        print("Load origin global model")
        self.load_model()

        for i in range(self.ul_rounds):
            s_t = time.time()
            self.selected_clients = self.select_target_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Flip Round number: {i + 1}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                print(f"-----------client {client.id} starts training----------")
                begin = time.time()
                if client.id in self.target_id:
                    client.fliptrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
                

            # 加权聚合,才是的数据量仍然为加权的数据量
            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            
            print("-------------After-------------")
            self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
        end = time.time()
        print('-' * 25, 'overall time cost', '-' * 25, end - begin)

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

            # # 保存每一轮的全局遗忘模型
            # self.save_ulandcon_model(i)
        print("\nEvaluate global model.")
        self.evaluate()
        print("\nBest test accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest target accuracy.")
        print(max(self.rs_target_acc))
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # 保存最终全局模型
        self.save_global_model()

    #待整理
    def ul_train(self):

        # print("Load origin global model")
        self.load_model()

        for i in range(self.ul_rounds):
            s_t = time.time()
            # self.selected_clients = self.select_clients()
            self.selected_clients = self.select_target_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                # print(f"\n-------------UL Round number: {i + 1}-------------")
                # print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                # print(f"-----------client {client.id} starts training----------")
                protos = client.test_protos()
                begin = time.time()
                if client.id in self.target_id:
                    client.ultrain()  # 目标客户端
                else:
                    client.train()  # 剩余客户端
                

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            print("-------------After-------------")
            self.evaluate()
            
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            
        end = time.time()
        print('-' * 25, 'total time', '-' * 25, end - begin)
            
            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        # print("\nEvaluate global model.")
        # self.evaluate()
        # print("\nBest test accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print("\nBest target accuracy.")
        # print(max(self.rs_target_acc))
        # # print("\nAverage time cost per round.")
        # # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


