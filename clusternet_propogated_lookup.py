import mxnet as mx
import numpy as np
from ResNet import resnet
import time
import copy
import logging
from sklearn.cluster import KMeans
import vgg
import gc

import time
import numpy as np



class ClusterNet(object):
    def __init__(self, in_prefix, in_epoch, batch_size, process=True, shrink = 8, dataset="cifar10", traindir=None, valdir=None, arch='resnet', lr = 0.00001):
        self.layers = []
        self.network = {}
        self.frozen_params=[]
        self.in_sym, self.in_args, self.in_auxs = mx.mod.module.load_checkpoint(in_prefix, in_epoch)


        for k in self.in_args:
            if "weight" in k:
                print k

        self.prefix=in_prefix
        self.epoch = in_epoch
        self.arch=arch
        self.shrink= shrink
        self.batch_size=batch_size
        self.lr = lr
        if dataset=='cifar10':
            self.val_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32), path_imgrec="dataset/cifar10_train.rec")
            auglist = mx.image.CreateAugmenter((3, 32, 32), resize=0, rand_mirror=True, hue=0.3, brightness=0.4,
                                               saturation=0.3, contrast=0.35, rand_crop=True, rand_gray=0.3)
            self.train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32),
                                            path_imgrec="dataset/cifar10_train.rec", aug_list=auglist)
        else:
            #self.val_iter = mx.io.ImageRecordIter(path_imgrec=imagenetpath, data_name="data", label_name="softmax_label",
            #    batch_size=batch_size, data_shape=(3, 224, 224))
            auglist = mx.image.CreateAugmenter((3, 224, 224), resize=0, rand_mirror=True, hue=0.4, brightness=0.4,
                                               saturation=0.4, contrast=0.4, rand_crop=True, rand_gray=0.4,rand_resize= True, pca_noise=0.2)
            auglist=mx.image.RandomOrderAug(auglist)

            self.train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 224, 224),
                                            path_imgrec=traindir, aug_list=auglist, shuffle=True)

            self.val_iter = mx.io.ImageRecordIter(path_imgrec=valdir, data_name="data", label_name="softmax_label",
                batch_size=batch_size, data_shape=(3, 224, 224))

            #firstbatch = tempiter.next()
            #nd_data = firstbatch.data
            #nd_label= firstbatch.label
            #self.val_iter = mx.io.NDArrayIter(nd_data,  label=nd_label,batch_size=batch_size)
            #del tempiter
            #gc.collect()


        self.input_shape = self.val_iter.provide_data[0][1]


        if process:
            if arch == "vgg":
                self.in_sym = vgg.get_symbol(1000, 16)
            self.sym, self.args = self.convert_network(self.in_sym, self.in_args, self.shrink)
            self.save_prefix=in_prefix+"_finetuned"
        else:
            self.sym=self.in_sym
            self.args=self.in_args
            self.save_prefix =in_prefix


    def get_quantized_filters(self, filters, shrink):
        shape = filters.shape
        n_clusters = shape[0] / shrink

        filters_shaped = filters.reshape((shape[0], shape[1] * shape[2] * shape[3]))
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit(filters_shaped.asnumpy())

        filter_kmean_indexes = estimator.predict(X=filters_shaped.asnumpy())

        return filter_kmean_indexes, mx.nd.array(estimator.cluster_centers_).reshape(n_clusters,shape[1],shape[2],shape[3])


    def propogate_lookup(self, in_filter, indices):
        if indices is not None:
            fshape = in_filter.shape
            filters_rearranged = mx.nd.zeros((fshape[0], fshape[1]/self.shrink, fshape[2], fshape[3]))
            for i, index in enumerate(indices):
                idx_casted = int(index)
                filters_rearranged[:, idx_casted, :] += in_filter[:, i, :]  #sum channels that would convolve the same codebook channel from previous layer
            return filters_rearranged
        else:
            return in_filter


    def convert_network(self,in_sym,in_args,shrink):
        print shrink
        layerlist = [elem for elem in in_sym.get_internals().list_outputs()  if "weight" in elem and "fc" not in elem]
        codebook_args = copy.deepcopy(self.in_args)
        previous_indices = None

        for element in layerlist[:-1]:

            weight = in_args[element]
            layer = element[:len(element) - 7]

            indices, codebook_filters = self.get_quantized_filters(weight, self.shrink)
            print "shape for layer: {} is {}".format(layer,weight.shape)
            looked_up_filters=self.propogate_lookup(codebook_filters, previous_indices)

            self.layers.append(layer)
            self.network[layer] = {}
            codebook_args[layer+"_weight"] = looked_up_filters

            previous_indices = indices

        lastlayer = layerlist[-1]
        lastweight= in_args[lastlayer]
        codebook_args[lastlayer] =  self.propogate_lookup(lastweight,previous_indices)
        last_convolution_layer=lastlayer[:len(lastlayer) - 7]



        def customconv(*args, **kwargs):
            kwargs['num_filter'] = kwargs['num_filter'] if kwargs['name']==last_convolution_layer else kwargs['num_filter']/shrink
            return mx.sym.Convolution(*args,no_bias=True ,**kwargs)


        if self.arch=='resnet':
            depth = 20
            per_unit = [(depth - 2) / 6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
            units = per_unit * 3
            return resnet(units=units, num_stage=3, filter_list=filter_list, num_class=10, data_type="cifar10",
                          bottle_neck=bottle_neck, custom_conv=customconv), codebook_args
        if self.arch=='vgg':
            return vgg.get_symbol(1000,16, batch_norm=False, conv_func = customconv), codebook_args
        else:
            print "this architecture is not supported yet"
            raise NotImplementedError



    def get_score(self,symbol_input, in_args, in_auxs, evalctx=mx.cpu()):
        mod = mx.mod.Module(symbol=symbol_input, context=evalctx)
        mod.bind(for_training=False, data_shapes=self.val_iter.provide_data, label_shapes=self.val_iter.provide_label)
        mod.set_params(in_args, in_auxs)
        begin = time.time()

        score= mod.score(self.val_iter, ['acc'])

        duration = time.time() - begin
        return score, duration

    def get_forward_time(self,symbol_input, in_args, in_auxs, evalctx=mx.cpu()):
        print evalctx
        mod = mx.mod.Module(symbol=symbol_input, context=evalctx)
        mod.bind(for_training=False, data_shapes=self.val_iter.provide_data, label_shapes=self.val_iter.provide_label)
        mod.set_params(in_args, in_auxs)
        begin = time.time()

        res= mod.predict(self.val_iter).asnumpy()

        duration = time.time() - begin
        return  duration

    def compare_baseline(self, evalctx=mx.cpu()):
        score2, time2 = self.get_score(self.sym, self.args, self.in_auxs, evalctx)
        score1, time1 = self.get_score(self.in_sym,self.in_args,self.in_auxs, evalctx)

        print "\nOriginal network score: {}, time to complete: {}".format(score1,time1)
        print "\nClustered network score: {}, time to complete: {}".format(score2,time2)
        speedup=float(time1)/time2
        print ("\nCompressed {}x, Speedup: {}".format(self.shrink,speedup))
        return speedup

    def compare_baseline2(self, evalctx=mx.cpu()):
        time1 = self.get_forward_time(self.in_sym,self.in_args,self.in_auxs, evalctx)
        time2 = self.get_forward_time(self.sym, self.args, self.in_auxs, evalctx)

        print "\nOriginal network score: time to complete: {}".format(time1)
        print "\nClustered network score: time to complete: {}".format(time2)
        speedup=float(time1)/time2
        print ("\nCompressed {}x, Speedup: {}".format(self.shrink,speedup))
        return speedup


    def finetune_codebooks(self):
        logging.getLogger().setLevel(logging.DEBUG)
        print"Finetuning codebook, frozen parameters: {}".format(self.frozen_params)
        mod = mx.mod.Module(symbol=self.sym, context=mx.gpu(),fixed_param_names=self.frozen_params)
        optimizer_params = {'learning_rate': self.lr,
                            'momentum': 0.9,
                            'wd': 0.0005,
                            'clip_gradient': None,
                            'rescale_grad': 1.0}
        epoch=self.epoch
        mod.fit(self.train_iter,
                eval_data=self.val_iter,
                optimizer='sgd',
                optimizer_params=optimizer_params,
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(self.batch_size, 150),
                epoch_end_callback=mx.callback.do_checkpoint(self.save_prefix),
                arg_params=self.args,
                aux_params=self.in_auxs,
                begin_epoch=epoch + 1,
                num_epoch=epoch + 51
                )