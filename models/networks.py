import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import sys
"""
Functions
##############################################################################
"""

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_network(which_model, num_inputs, norm='batch', use_dropout=False, gpu_ids=[]):
    net = None
    use_gpu = len(gpu_ids) > 0
    #norm_layer = get_norm_layer(norm_type=norm)
    num_classes = 2
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif which_model == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif which_model == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif which_model == 'vgg_sara':
        from models.vgg import vgg11_bn_me
        net = vgg11_bn_me(num_inputs)
    elif which_model == 'vgg_sara_small':
        from models.vgg import vgg9_bn
        net = vgg9_bn(num_inputs)
    elif which_model == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif which_model == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif which_model == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif which_model == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif which_model == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif which_model == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif which_model == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif which_model == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif which_model == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif which_model == 'xception':
        from models.xception import xception
        net = xception()
    elif which_model == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif which_model == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif which_model == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif which_model == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif which_model == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif which_model == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif which_model == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif which_model == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif which_model == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif which_model == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif which_model == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif which_model == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif which_model == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif which_model == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif which_model == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif which_model == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif which_model == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif which_model == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif which_model == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif which_model == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif which_model == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif which_model == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif which_model == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif which_model == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif which_model == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif which_model == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif which_model == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif which_model == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif which_model == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif which_model == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif which_model == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network _%s_ you have entered is not supported yet'%which_model)
        sys.exit()

    if len(gpu_ids) > 0:
        net.cuda(device=gpu_ids[0])

    net.apply(weights_init)

    if use_gpu: #use_gpu
        net = net.cuda()

    return net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
