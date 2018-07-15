import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import AlexNet
import math
from sklearn.cluster import KMeans
import time
import gc
import Imagenet
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class clusternet(object):
    def __init__(self, train_dir, val_dir, shrink=2):
        self.in_net     = AlexNet.AlexNet()
        self.in_params  = self.in_net.state_dict()
        self.shrink     = shrink
        self.imagenet_dummy=torch.rand(64,3,224,224)

        self.train_path=train_dir
        self.val_path=val_dir
        self.batch_size=32


    def evaluate(self, evaldir, batch_size, quantized=False):

        net = self.convert_network() if quantized else self.in_net

        valloader = Imagenet.load_imagenet(evaldir, batch_size)
        print Imagenet.validate(valloader, net, nn.CrossEntropyLoss())


    def convert_network(self):
        grouped_layers = set(['4', '10', '12'])
        indices, codebooks = self.quantize_all_params(grouped_layers)

        codebooks=iter(codebooks)

        qnet=AlexNet.AlexNetLookup(indices, shrink=self.shrink)
        qnet_params=qnet.state_dict()

        for k in qnet_params:
            if "weight" in k and "features" in k:
                qnet_params[k].data.copy_(next(codebooks))

        return qnet





    def get_quantized_filters(self, filters, shrink, groups= 1):

        shape = filters.shape
        chunksize = shape[0] / groups
        n_clusters_total = shape[0] / shrink


        filters_shaped = filters.reshape((shape[0], shape[1] * shape[2] * shape[3])).data.numpy()

        indices=[]
        codebooks=[]

        for group_idx in range(groups):
            filter_group_idx = group_idx*chunksize
            filter_group=filters_shaped[filter_group_idx:filter_group_idx+chunksize]

            n_clusters = filter_group.shape[0] / shrink

            estimator = KMeans(n_clusters=n_clusters)
            estimator.fit(filter_group)
            ind = estimator.predict(X=filter_group)

            indices.append(ind + group_idx*n_clusters)
            codebooks.append(estimator.cluster_centers_)

#
        return torch.LongTensor(np.concatenate(indices)), torch.Tensor(np.concatenate(codebooks)).reshape(n_clusters_total,shape[1],shape[2],shape[3])


    def quantize_all_params(self, grouped_layers = set([]), debug=False):
        in_params = self.in_params
        indices_list = []
        codebooks = []


        for k in in_params:
            if "weight" in k and "features" in k:
                filters = in_params[k]
                groups = 2 if k.split('.')[1] in grouped_layers else 1
                indices, codebook = self.get_quantized_filters(filters, self.shrink, groups=groups)

                indices_list.append(indices)
                codebooks.append(codebook)


        return indices_list, codebooks


    def forward_pass(self, net, data):
        return net(data).data.numpy()



    def sanity_test2(self):

        begin=time.time()
        res1=self.forward_pass(self.in_net, self.imagenet_dummy)
        time1=time.time()-begin


        qnet = self.convert_network()

        del self.in_net
        del self.in_params
        gc.collect()

        begin=time.time()
        res2=self.forward_pass(qnet, self.imagenet_dummy)
        time2=time.time()-begin
        err = np.mean(np.square(res1-res2))

        print "speedup is {}, mean err is {}".format(float(time1)/time2, err)




#cn=clusternet(shrink=4)
#
#cn.sanity_test2()