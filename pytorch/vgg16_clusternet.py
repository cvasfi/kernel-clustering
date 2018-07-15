import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math
from sklearn.cluster import KMeans
import time
from clusternet import clusternet
import vgg16
from vgg16 import VGG

class vgg16_clusternet(clusternet):
    def __init__(self, weights, shrink=2):
        self.cfg        = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.in_net     = VGG(vgg16.make_layers(self.cfg))
        self.in_net.load_state_dict(torch.load(weights))
        self.in_params  = self.in_net.state_dict()
        self.shrink     = shrink
        self.imagenet_dummy=torch.rand(8,3,224,224)


    def convert_network(self):

        indices, codebooks = self.quantize_all_params()
        codebooks=iter(codebooks)


        cfg_lookup = [n / self.shrink if isinstance(n, int) else n for n in self.cfg]

        qnet=VGG(vgg16.make_layers_lookup(cfg_lookup, iter(indices), self.shrink))
        qnet_params=qnet.state_dict()

        for k in qnet_params:
            if "weight" in k and "features" in k:
                qnet_params[k].data.copy_(next(codebooks))

        return qnet






cn=vgg16_clusternet(weights="/home/tapir/Documents/Thesis/vgg16_pytoarch_weights/vgg16-397923af.pth", shrink=4)

cn.sanity_test2()
