import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.nn import CrossEntropyLoss, BCELoss
import sys
from torch import nn

class CNNModel(BaseModel):
    def name(self):
        return 'CNNModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.roisize
        
        self.opt = opt
        self.netName = opt.which_model
        
        # change number of channels based on how many input types we have
        
        self.input = self.Tensor(nb, len(opt.input_seq), size*2, size*2)
        
        if self.opt.phase == 'train':
            if self.opt.class_weight:
                print('***** training with weighted samples *****')
        self.label = Variable()
        self.loss = 0
        self.net = networks.define_network(opt.which_model, len(opt.input_seq), opt.norm, not opt.no_dropout, self.gpu_ids).cuda()
        self.net = self.net.float()
            
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            print('loading existing model:',self.netName, '-', which_epoch)
            self.load_network(self.net, self.netName, which_epoch)
            
        if self.isTrain:
            self.old_lr = opt.lr
            self.imagepool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterion = BCELoss()
            
            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)                     
            #self.optimizer = torch.optim.SGD(self.net.parameters(),lr=opt.lr, momentum=0.9, weight_decay=0.1)
            
        print('-- Networks initialized')      
        if self.isTrain:
            networks.print_network(self.net)
            
    def set_input(self, input):
        
        self.input = Variable(input['tensor'].cuda())
        self.label = Variable(input['label'].float())
        
    def forward(self):
        
        #m = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        self.pred = self.net.forward(self.input.detach()).cpu()
        self.pred = m(self.pred)

    def test(self):
        #m = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        self.pred = self.net.forward(self.input.detach()).cpu()
        self.pred = m(self.pred)
        return self.pred

    def backward(self):
        
        #self.net.forward(self.input.detach())
        self.loss = self.criterion(self.pred, self.label)  
        
        if self.opt.class_weight:# add weights to samples with lower abundance
            
            if self.opt.target == 'SOX2':
                # we have twice more positive cases for sox2, so more weight for NEGATIVE sample
                if torch.equal(self.label, torch.tensor([[1., 0.]])):  
                    beta = 2
                    self.loss = beta * self.loss

            if self.opt.target == 'Ki67':
                # we have more twice more negative cases for ki67, so more weight for POSITIVE sample
                if torch.equal(self.label, torch.tensor([[0., 1.]])): 
                    
                    beta = 2 
                    self.loss = beta * self.loss
                
        self.loss.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer.zero_grad() # clearing the gradients of the model parameters
        self.backward() # computing the updated weights of the model
        self.optimizer.step()
        
    def get_current_errors(self):
        
        #return [self.Lloss, self.Uloss, self.loss]
        return [self.loss]
    
    def save(self, label):
        self.save_network(self.net, self.netName, label, self.gpu_ids)
    
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
                                              
    def print_model():
        self.net.print_network()
    
    def get_net(self):
        return self.net
                                              

