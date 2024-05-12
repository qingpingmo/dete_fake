import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, lnum=opt.lnum)
            # self.model.fc = nn.Linear(2048, 1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad Ture',name)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if not pa.requires_grad: print('='*20, 'requires_grad False',name)
        print('='*30)
        net_params = sum(map(lambda x: x.numel(), self.model.parameters()))
        print(f'Model parameters {net_params:,d}')
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                # self.optimizer = torch.optim.Adam(self.model.parameters(),
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        # self.model.to(opt.gpu_ids[0])
        self.model = nn.DataParallel(self.model).cuda()


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.delr
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/self.delr} to {param_group["lr"]} with delr {self.delr}')
        print('*'*25)
        return True

    def set_input(self, input):
        # self.input = input[0].to(self.device)
        # self.label = input[1].to(self.device).float()
        self.input = input[0].cuda()
        self.label = input[1].cuda().float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

