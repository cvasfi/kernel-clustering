import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import AlexNet
import math
from sklearn.cluster import KMeans
import time

class clusternet(object):
    def __init__(self, modelpath=None, shrink=2):
        self.in_net     = AlexNet.AlexNet()
        self.in_params  = self.in_net.state_dict()
        self.shrink     = shrink
        self.imagenet_dummy=torch.rand(64,3,224,224)




    def convert_network(self):

        indices, codebooks = self.quantize_all_params()
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

            indices.append(estimator.predict(X=filter_group) + group_idx*n_clusters)
            codebooks.append(estimator.cluster_centers_)


        return torch.LongTensor(np.concatenate(indices)), torch.Tensor(np.concatenate(codebooks)).reshape(n_clusters_total,shape[1],shape[2],shape[3])


    def quantize_all_params(self, debug=False):
        in_params = self.in_params
        indices_list = []
        codebooks = []

        for k in in_params:
            if "weight" in k and "features" in k:
                filters = in_params[k]
                indices, codebook = self.get_quantized_filters(filters, self.shrink)

                indices_list.append(indices)
                codebooks.append(codebook)

                #if debug:
                #    in_params[k].data.copy_(q_filters)
                #    assert torch.equal(in_params[k], q_filters)

        return indices_list, codebooks


    def forward_pass(self, net, data):
        return net(data)

    #def sanity_test1(self):
    #    res1=self.forward_pass(self.in_net, self.imagenet_dummy)
    #    self.quantize_all_params(debug=True)
#
    #    res2=self.forward_pass(self.in_net, self.imagenet_dummy)
    #    err = torch.mean((res1 - res2) ** 2)
#
    #    print err.data.numpy()


    def sanity_test2(self):
        begin=time.time()
        res1=self.forward_pass(self.in_net, self.imagenet_dummy)
        time1=time.time()-begin

        qnet = self.convert_network()

        begin=time.time()
        res2=self.forward_pass(qnet, self.imagenet_dummy)
        time2=time.time()-begin
        err = torch.mean((res1 - res2) ** 2)

        print "speedup is {}, mean err is {}".format(float(time1)/time2, err.data.numpy())


cn=clusternet("/home/tapir/Documents/Thesis/kernel-clustering/notebooks/test.pt",shrink=4)

cn.sanity_test2()