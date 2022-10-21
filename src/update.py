#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):  # , logger
        self.args = args
        # self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = format((correct/total) *100, '.2f')
        return accuracy, loss

class DataFreeDistillation(object):
    def __init__(self, args, dataset, idxs):  # dataset=test dataset ; idx也是测试集的下标
        self.args = args
        self.testloader = DataLoader(DatasetSplit(dataset, idxs),
                                batch_size=int(len(idxs)/10), shuffle=False)
        self.device = 'cuda' if args.gpu else 'cpu'

        # 这里得看一下蒸馏的setting
        self.criterion = nn.NLLLoss().to(self.device)


    def distillation(self, student, generator, teacher, global_round):
        # transmit model to device
        student = student.to(self.device)
        teacher = teacher.to(self.device)
        generator = generator.to(self.device)
        teacher.eval()
        student.train()
        generator.train()

        # Set optimizer for the student model and generator
        optimizer_S = torch.optim.SGD(student.parameters(), lr=self.args.lr_S, weight_decay=self.args.weight_decay, momentum=0.9 )
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr_G )


        if self.args.scheduler:
            scheduler_S = torch.optim.lr_scheduler.MultiStepLR(optimizer_S, [100, 200], 0.1)
            scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], 0.1)
        
        # 下面开始实现对抗蒸馏  
        epoch_loss = []
        for iter in range(self.args.local_ep):  # 局部对抗训练的轮数 100
            batch_loss = []
            for k in range(self.args.iter_discrim):  # 每一轮更新学生模型的轮数 5
                z = torch.randn((self.args.local_bs, self.args.nz, 1, 1) ).to(self.device)
                optimizer_S.zero_grad()
                fake = generator(z).detach()
                t_logit = teacher(fake)
                s_logit = student(fake)
                loss_S = torch.nn.functional.F.l1_loss(s_logit, t_logit.detach())
            
                loss_S.backward()
                optimizer_S.step()
                batch_loss.append(loss_S.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            G_loss = 0.
            for k in range(self.args.iter_gen):  # 每一轮更新生成器的轮数
                z = torch.randn((self.args.local_bs, self.args.nz, 1, 1) ).to(self.device)
                optimizer_G.zero_grad()
                generator.train()
                fake = generator(z)
                t_logit = teacher(fake) 
                s_logit = student(fake)

                #loss_G = - torch.log( F.l1_loss( s_logit, t_logit )+1) 
                loss_G = - torch.nn.functional.F.l1_loss( s_logit, t_logit ) 
                G_loss = loss_G.item()

                loss_G.backward()
                optimizer_G.step()
            
            print('| Global Round : {} | Local Epoch : {}/{} ({:.0f}%)]\t| S_Loss: {:.6f}   G_LOSS:{:.6f}'.format(
                        global_round, iter, self.args.local_ep,
                        100. * iter / self.args.local_ep, G_loss, sum(batch_loss)/len(batch_loss)))

        return student.state_dict(), generator.state_dict(), sum(epoch_loss) / len(epoch_loss)

        





        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.trainloader):
        #         images, labels = images.to(self.device), labels.to(self.device)

        #         model.zero_grad()
        #         log_probs = model(images)
        #         loss = self.criterion(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         if self.args.verbose and (batch_idx % 10 == 0):
        #             print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 global_round, iter, batch_idx * len(images),
        #                 len(self.trainloader.dataset),
        #                 100. * batch_idx / len(self.trainloader), loss.item()))
        #         # self.logger.add_scalar('loss', loss.item())
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))



def test_inference(args, model, test_dataset, idxs):  # 这个应该可以复用，测local model还是global model都行
    """ Returns the test accuracy and loss.
    """

    

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, idxs),
                                 batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = format((correct/total) * 100, '.2f')
    return accuracy, loss
