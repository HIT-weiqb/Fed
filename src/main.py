#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import random
from gan import GeneratorA
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import PreTrained, DataFreeDistillation, test_inference
from models import DNN, CNN1, CNN2, CNN3, LeNet, AlexNet, ResNet18, ResNet34, mobilenetv2, shufflenetv2
from utils import get_dataset, average_weights, exp_details


CANDIDATE_MODELS = {"DNN": DNN,   ## 一共十个模型架构
                    "CNN1": CNN1,
                    "CNN2": CNN2,
                    "CNN3": CNN3,
                    "LeNet": LeNet,
                    "AlexNet": AlexNet,
                    "shufflenetv2": shufflenetv2,
                    "mobilenetv2": mobilenetv2,
                    "ResNet18": ResNet18,
                    "ResNet34": ResNet34
                    } 
MODEL_NAMES = ["DNN", "CNN1", "CNN2", "CNN3", "LeNet", "AlexNet", "shufflenetv2", "mobilenetv2", "ResNet18", "ResNet34"]
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = '/home/aiia611/wqb/data'  # /data_b/wqb/src/data
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # 设置随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.gpu_id>=0:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu is not None else 'cpu'
    # load dataset and user groups
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(args, path_project)

    # BUILD MODEL
    # if args.dataset == 'cifar':
    #     global_model = CNNCifar(args=args)
    Pretrained_Models = {}
    for i in range(args.num_users):
        Pretrained_Models[i] = CANDIDATE_MODELS[MODEL_NAMES[i]](args)

    # Set the model to train and send it to device.
    for i in range(args.num_users):
        Pretrained_Models[i].to(device)
        Pretrained_Models[i].train()
        # print(Pretrained_Models[i])
    # global_model.to(device)
    # global_model.train()
    # print(global_model)

    if args.pretrained == 1:  # 预训练好了，加载数据集的划分、准确率、模型参数
        model_path = os.path.join('{}/models'.format(path_project), 'checkpoint_{}_iid[{}].pth.tar'.format(args.dataset, args.iid))
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        user_groups = checkpoint['user_groups']
        user_groups_test = checkpoint['user_groups_test']
        list_test_acc = checkpoint['test_accuracy'] 
        list_test_loss = checkpoint['test_loss']
        
        print('Pre-trained Model Loaded.\n')
        for idx in range(args.num_users):
            Pretrained_Models[idx].load_state_dict(checkpoint[MODEL_NAMES[idx]])
            print('| Client Idx : {} | Model Architecture : {} | Test Acc : {}  Test loss : {}'.format(idx, MODEL_NAMES[idx], list_test_acc[idx], list_test_loss[idx])) 
    else:
        # Training
        train_loss, train_accuracy = [], []


        for idx in range(args.num_users):
        # 对每个局部模型做预训练
            print('#################################################################################')
            print('Training Clinet Idx: {} , Model Architecutre : {}'.format(idx, MODEL_NAMES[idx]))

            pretrained = PreTrained(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx])  # , logger=logger
            model = Pretrained_Models[idx]

            # Set optimizer for the local updates
            if args.pretrained_optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrained_lr,
                                            momentum=args.pretrained_momentum)
            elif args.pretrained_optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrained_lr,
                                            weight_decay=1e-4)

            for epoch in tqdm(range(args.pretrained_epochs)):
                w, loss = pretrained.update_weights(model=model, optimizer=optimizer, global_round=epoch+1, idx=idx)
                model.load_state_dict(w)  # 将训练的模型参数保存下来

                if (epoch+1) % 10 == 0:
                    # eval
                    list_acc, list_loss = [], []
                    acc, loss = pretrained.inference(model=model)            
                    print('##############################################################################################')
                    print(' Client Idx: {} | Model Architecture: {} | Training Round: {} | Eval Acc: {}   Eval Loss: {} |'.format(
                        idx, MODEL_NAMES[idx], epoch+1, acc, loss))
                    print('##############################################################################################')

        # Test inference after completion of pretraining
        list_test_acc, list_test_loss = [], []
        for idx in range(args.num_users):
            test_acc, test_loss = test_inference(args, Pretrained_Models[idx], test_dataset, user_groups_test[idx])
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)

        print('Final Test Loss : ' ,list_test_loss)
        print('Final Test Accuracy: ' ,list_test_acc)

        # # Saving the objects train_loss and train_accuracy:
        # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
        #            args.local_ep, args.local_bs)

        # with open(file_name, 'wb') as f:
        #     pickle.dump([train_loss, train_accuracy], f)

        checkpoint = {}  # 存储已预训练好的模型
        checkpoint['test_accuracy'] = list_test_acc
        checkpoint['test_loss'] = list_test_loss
        checkpoint['user_groups'] = user_groups
        checkpoint['user_groups_test'] = user_groups_test
        for idx in range(args.num_users):
            checkpoint[MODEL_NAMES[idx]] = Pretrained_Models[idx].state_dict()
        model_path = os.path.join('{}/models'.format(path_project),'checkpoint_{}_iid[{}].pth.tar'.format(args.dataset, args.iid))
        torch.save(checkpoint, model_path)



    # federated learning & data free distillation
    # global model & generator
    global_model = CANDIDATE_MODELS[args.model](args)  
    global_model.to(device)
    global_model.train()
    print(global_model)
    generator = GeneratorA(nz=args.nz, nc=3, img_size=32)
    local_gen_user = [generator.state_dict() for i in range(args.num_users)]  # 一个列表 里面存储了所有client的generator的参数

    # copy weights 存储全局模型的参数
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []

    for epoch in tqdm(range(args.comm_rounds)):  # 通讯轮数
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        for idx in range(args.num_users):  # 用deepcopy来实现，global model初始化local model
            generator.load_state_dict(local_gen_user[idx])  # 加载local generator
            local_model = DataFreeDistillation(args=args, dataset=test_dataset,
                                      idxs=user_groups_test[idx])  # , logger=logger
            w, w_gen, loss = local_model.distillation(  # 在这里实现蒸馏
                model=copy.deepcopy(global_model), generator=generator, teacher=Pretrained_Models[idx], global_round=epoch, client=idx)
        
            # record local generator
            local_gen_user[idx] = w_gen
            local_weights.append(copy.deepcopy(w))  # 统计各client的local model参数
            local_losses.append(copy.deepcopy(loss))

            # 测试在local dataset上的准确率和loss
            if epoch % 5 == 0:
                tmp_weights = global_model.state_dict()  # 暂存global model的权重
                global_model.load_state_dict(w)  # 加载local model

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)


        loss_avg = sum(local_losses) / len(local_losses)  # 这一通讯轮次的平均loss
        train_loss.append(loss_avg)

        if epoch % 10 == 0:  # 测试global model在整个dataset上的性能
            t=1
    # 还差最后总的，对global model的测试
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
