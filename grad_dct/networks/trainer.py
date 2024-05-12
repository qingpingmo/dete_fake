import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from torch.autograd import Variable
import numpy as np

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # self.model = resnet50(pretrained=True)
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True
    # def adjust_learning_rate(self, min_lr=1e-6):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] /= 10.
    #         if param_group['lr'] < min_lr:
    #             return False
    #     return True

    def set_input(self, input_d1, input_d2):
        self.input_d1 = input_d1[0].to(self.device)
        self.input_d2 = input_d2[0].to(self.device)
        self.label_d1 = input_d1[1].to(self.device).float()
        self.label_d2 = input_d2[1].to(self.device).float()


    def forward_d1(self):
        self.output_d1_cls, self.output_d1_domain = self.model.forward_train(self.input_d1, self.label_d1)
    
    def forward_d2(self):
        self.output_d2_cls, self.output_d2_domain = self.model.forward_train(self.input_d2, self.label_d2)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)
    def get_domain_loss(self):
        return self.loss_fn(self.output_domain, self.label)

    def optimize_parameters_baseline(self):
        self.forward_d1()
        self.forward_d2()
        self.loss_d1_cls = self.loss_fn(self.output_d1_cls.squeeze(1), self.label_d1)
        self.loss_d1_domain = self.loss_fn(self.output_d1_domain.squeeze(1), torch.zeros_like(self.output_d1_domain.squeeze(1)).float().cuda())
        self.loss_d2_cls = self.loss_fn(self.output_d2_cls.squeeze(1), self.label_d2)
        self.loss_d2_domain = self.loss_fn(self.output_d2_domain.squeeze(1), torch.ones_like(self.output_d2_domain.squeeze(1)).float().cuda())
        self.loss = self.loss_d1_cls + self.loss_d2_cls
        print("loss_d1_cls: {} loss_d2_cls: {}".format(self.loss_d1_cls, self.loss_d2_cls))
        # self.loss = self.loss_d1_cls + self.loss_d1_domain + self.loss_d2_cls + self.loss_d2_domain
        # print("loss_d1_cls: {} loss_d1_domain: {} loss_d2_cls: {} loss_d2_domain: {}".format(self.loss_d1_cls, self.loss_d1_domain, self.loss_d2_cls, self.loss_d2_domain))
        # import pdb; pdb.set_trace()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        #################3
        #self.loss = self.loss_d1_cls + self.loss_d1_domain + self.loss_d2_cls + self.loss_d2_domain
        #self.loss = cls_loss_weight * (self.loss_d1_cls + self.loss_d2_cls) + domain_loss_weight * (self.loss_d1_domain + self.loss_d2_domain)

        #print("loss_d1_cls: {} loss_d1_domain: {} loss_d2_cls: {} loss_d2_domain: {}".format(self.loss_d1_cls, self.loss_d1_domain, self.loss_d2_cls, self.loss_d2_domain))
        #####################3


    def optimize_parameters_ours(self):
        cls_loss_weight = 0.9
        domain_loss_weight = 0.1

        self.forward_d1()
        self.forward_d2()

        def safe_squeeze(tensor):
            if tensor.dim() > 1:
                return tensor.squeeze(1)
            return tensor

        self.loss_d1_cls = self.loss_fn(safe_squeeze(self.output_d1_cls), self.label_d1)
        self.loss_d1_domain = self.loss_fn(safe_squeeze(self.output_d1_domain), torch.zeros_like(safe_squeeze(self.output_d1_domain)).float().cuda())

        self.loss_d2_cls = self.loss_fn(safe_squeeze(self.output_d2_cls), self.label_d2)
        self.loss_d2_domain = self.loss_fn(safe_squeeze(self.output_d2_domain), torch.ones_like(safe_squeeze(self.output_d2_domain)).float().cuda())

    
        lambda_value = (self.loss_d1_domain + self.loss_d2_domain) / (self.loss_d1_cls + self.loss_d2_cls)
    
    
        if torch.isnan(lambda_value).any():
            lambda_value = torch.tensor(1e7)

    
        lambda_value_int = int(lambda_value)
    
        if lambda_value_int == 0:
            lambda_value_int = 1
    
        self.loss = cls_loss_weight * (self.loss_d1_cls + self.loss_d2_cls) + domain_loss_weight * (1 / lambda_value_int) * (self.loss_d1_domain + self.loss_d2_domain)
    
        print("loss_d1_cls: {} loss_d1_domain: {} loss_d2_cls: {} loss_d2_domain: {}".format(self.loss_d1_cls, self.loss_d1_domain, self.loss_d2_cls, self.loss_d2_domain))
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
