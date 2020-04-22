import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import glob
from collections import OrderedDict
import multiprocessing
from torch import nn
import time

import argparse

from model import AlexNet
from misc import progress_bar
from tensorboardX import SummaryWriter

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.is_board = False

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='/mnt/disk50/datasets/cifar', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='/mnt/disk50/datasets/cifar', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model_from_pth(self, model_path):
        """Load the pre-trained model weight

        :param model_path:
        :return:
        """
        checkpoint = torch.load(model_path, map_location=self.device_name)['model']

        # TODO:这里需要具体了解原因在哪里?
        checkpoint_parameter_name = list(checkpoint.keys())[0]
        model_parameter_name = next(self.model.named_parameters())[0]

        is_checkpoint = checkpoint_parameter_name.startswith('module.')
        is_model = model_parameter_name.startswith('module.')

        if is_checkpoint and not is_model:
            # 移除checkpoint模型里面参数
            new_parameter_check = OrderedDict()
            for key, value in checkpoint.items():
                if key.startswith('module.'):
                    new_parameter_check[key[7:]] = value
            self.model.load_state_dict(new_parameter_check)
        elif not is_checkpoint and is_model:
            # 添加module.参数
            new_parameter_dict = OrderedDict()
            for key, value in checkpoint.items():
                if not key.startswith('module.'):
                    key = 'module.' + key
                    new_parameter_dict[key] = value
        else:
            self.model.load_state_dict(checkpoint)
        return self.model

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda:0')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        self.model = AlexNet().to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self, writer=None):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        # if not writer:
        #     writer.add_scalar

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        start = time.time()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        end = time.time()
        time_used = end - start

        return test_loss, test_correct / total, time_used

    def save(self):
        model_out_path = "./best_model_new.pkl"
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        # for k, v in self.model.state_dict():
        #     print('layer{}'.k)
        #     print(v)
        accuracy = 0
        writer = SummaryWriter()
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            # writer.add_scalar('loss_group',{'train_loss':train_loss.numpy(),
            #                                 'test_loss':test_loss.numpy()},epoch)
            # writer.add_scalar('acc_group',{'train_acc':train_acc.numpy(),
            #                                'test_acc':test_acc.numpy()}, epoch)

            if test_acc > accuracy:
                accuracy = test_acc
                self.save()
            elif epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()

