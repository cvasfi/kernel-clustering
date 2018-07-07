import mxnet as mx
import numpy as np
from ResNet import resnet
import time
import copy
import logging
from sklearn.cluster import KMeans

import AlexNet

import time
import numpy as np




class converter(object):

    def __init__(self, in_prefix,in_epoch, batch_size, data_path = "dataset/cifar10_val.rec", process=True, shrink = 8, arch="resnet"):
        self.arch=arch
        self.layers = []
        self.network = {}
        self.frozen_params=[]
        self.in_sym, self.in_args, self.in_auxs = mx.mod.module.load_checkpoint(in_prefix, in_epoch)
        self.prefix=in_prefix
        self.epoch = in_epoch
        self.batch_size = batch_size
        self.shrink=shrink
        if arch=='resnet':
            self.val_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32), path_imgrec=data_path)
            auglist = mx.image.CreateAugmenter((3, 32, 32), resize=0, rand_mirror=True, hue=0.3, brightness=0.4,
                                               saturation=0.3, contrast=0.35, rand_crop=True, rand_gray=0.3)
            self.train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32),
                                            path_imgrec="dataset/cifar10_train.rec", aug_list=auglist)
        else:
            dummysize=1000
            dummy_data=mx.nd.random_uniform(shape=(dummysize,3,224,224))
            dummy_labels=mx.nd.array(np.random.choice(1000,dummysize))
            self.val_iter = mx.io.NDArrayIter(data=dummy_data,label=dummy_labels,batch_size=batch_size)

        self.input_shape = self.val_iter.provide_data[0][1]
        self.codebook_args = copy.deepcopy(self.in_args)

        if process:
            self.process_symbol()
            self.sym = self.convert(arch)
            self.args= self.codebook_args
            self.save_prefix=in_prefix+"_finetuned"
        else:
            self.sym=self.in_sym
            self.frozen_params = [n for n in self.sym.get_internals().list_outputs() if "indices" in n]
            self.args=self.in_args
            self.save_prefix =in_prefix



    def get_quantized_filters(self, filters, shrink):
        shape = filters.shape
        n_clusters = shape[0] / shrink

        filters_shaped = filters.reshape((shape[0], shape[1] * shape[2] * shape[3]))
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit(filters_shaped.asnumpy())

        filter_kmean_indexes = estimator.predict(X=filters_shaped.asnumpy())
        filters_quantized = np.array([estimator.cluster_centers_[idx] for idx in filter_kmean_indexes])

        return mx.nd.array(filter_kmean_indexes), mx.nd.array(estimator.cluster_centers_).reshape(n_clusters,shape[1],shape[2],shape[3]), mx.nd.array(filters_quantized).reshape(shape)


    def get_onehot(self, data, nclusters, batch_size):
        index_mat = mx.nd.one_hot(data, depth=nclusters).reshape(0, -1)
        return mx.nd.broadcast_axes(mx.nd.expand_dims(index_mat, axis=0), axis=0, size=batch_size)



    def process_symbol(self):
        print "processing symbol"
        in_shape=self.input_shape
        in_sym  =  self.in_sym
        in_args = self.in_args

        for element in in_sym.get_internals().list_outputs():
            if "weight" in element and "fc" not in element:  # maybe tune for different networks

                weight = in_args[element]
                layer = element[:len(element) - 7]

                indices, codebook_filters, quantized_filters = self.get_quantized_filters(weight, self.shrink)
                lrshape = in_sym.get_internals()[layer + "_output"].infer_shape(data=in_shape)[1]
                #codebook_flattened =   mx.nd.transpose(mx.nd.array(codebook_filter), axes=(1,0,2,3)).reshape((-1,1,0, 0)) # TODO: put this in codebook extraction function
                onehot_indices = self.get_onehot(indices,indices.shape[0]/self.shrink,self.batch_size)
                self.layers.append(layer)
                self.network[layer] = {}
                self.codebook_args[layer+"_weight"] = codebook_filters
                self.codebook_args[layer + "_indices"] = onehot_indices
                self.frozen_params.append(layer + "_indices")
                self.network[layer]["f_shape"] = codebook_filters.shape
                self.network[layer]["i_shape"] = onehot_indices.shape
                self.network[layer]["out_shape"] = self.get_int_shape(lrshape[0])


        print "symbol processed, clusters extracted."

    def get_int_shape(self,tpl):
        return (int(tpl[0]),int(tpl[1]),int(tpl[2]),int(tpl[3]))    #i know

    def clustered_convolution(self, data, name, num_filter = None, kernel=(3,3), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace = None):
        layer = self.network[name]
        fshape = layer["f_shape"]
        indices_shape = layer["i_shape"]
        output_shape = layer["out_shape"]
        filters = mx.sym.Variable(name+"_weight", shape=fshape)
        indices = mx.sym.Variable(name+"_indices", shape=indices_shape)

        res = mx.sym.Convolution(data=data, weight=filters, num_filter=fshape[0], stride = stride,
                                 no_bias=no_bias, kernel=kernel, pad=pad, name=name)

        res= mx.sym.batch_dot(indices, res.reshape((0, 0, -1))).reshape(output_shape)

        return res

    def convert(self, arch):
        if arch=='resnet':
            depth = 20
            per_unit = [(depth - 2) / 6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
            units = per_unit * 3
            return resnet(units=units, num_stage=3, filter_list=filter_list, num_class=10, data_type="cifar10",
                         bottle_neck=bottle_neck, custom_conv=self.clustered_convolution)
        else:
            return AlexNet.get_symbol(1000,self.clustered_convolution)

    def get_score(self,symbol_input, in_args, in_auxs, evalctx=mx.cpu()):
        print evalctx
        mod = mx.mod.Module(symbol=symbol_input, context=evalctx)
        mod.bind(for_training=False, data_shapes=self.val_iter.provide_data, label_shapes=self.val_iter.provide_label)
        mod.set_params(in_args, in_auxs)
        begin = time.time()

        score= mod.score(self.val_iter, ['acc'])

        duration = time.time() - begin
        return score, duration

    def evaluate_converted(self):
        return self.get_score(self.sym, self.args, self.in_auxs)

    def compare_baseline(self, evalctx=mx.cpu()):
        score1, time1 = self.get_score(self.in_sym,self.in_args,self.in_auxs, evalctx)
        score2, time2 = self.get_score(self.sym, self.args, self.in_auxs, evalctx)

        print "\nOriginal network score: {}, time to complete: {}".format(score1,time1)
        print "\nClustered network score: {}, time to complete: {}".format(score2,time2)
        speedup=float(time1)/time2
        print ("\nCompressed {}x, Speedup: {}".format(self.shrink,speedup))
        return speedup


    def finetune_codebooks(self):
        logging.getLogger().setLevel(logging.DEBUG)
        print"Finetuning codebook, frozen parameters: {}".format(self.frozen_params)
        mod = mx.mod.Module(symbol=self.sym, context=mx.gpu(),fixed_param_names=self.frozen_params)
        optimizer_params = {'learning_rate': 0.00001,
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




def train(in_prefix, in_epoch):
    logging.getLogger().setLevel(logging.DEBUG)

    auglist = mx.image.CreateAugmenter((3, 32, 32), resize=0, rand_mirror=True, hue=0.3, brightness=0.4,
                                       saturation=0.3, contrast=0.35, rand_crop=True, rand_gray=0.3)
    batch_size = 32
    train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32),
                                         path_imgrec="dataset/cifar10_train.rec", aug_list=auglist)
    val_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32), path_imgrec="dataset/cifar10_val.rec")

    sym, args, auxs = mx.mod.module.load_checkpoint(in_prefix, in_epoch)

    mod = mx.mod.Module(symbol=sym, context=mx.gpu())
    optimizer_params = {'learning_rate': 0.0001,
                        'momentum': 0.9,
                        'wd': 0.0005,
                        'clip_gradient': None,
                        'rescale_grad': 1.0}
    epoch = in_epoch
    mod.fit(train_iter,
            eval_data=val_iter,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            eval_metric='acc',
            batch_end_callback=mx.callback.Speedometer(batch_size, 150),
            epoch_end_callback=mx.callback.do_checkpoint(prefix),
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch + 1,
            num_epoch=epoch + 51
            )

#
#prefix="cnn_models/filter_level2x/resnet20"
#epoch=124
prefix="cnn_models/alexnet/alexnet_test"
epoch=1

speedup_sum=0
loop=5
cv = converter(prefix, epoch, batch_size=128, process=True, shrink=4, arch="alexnet")
cv.compare_baseline(mx.cpu())

#for i in range(loop):
#    #cv.convert()
#    speedup_sum+=cv.compare_baseline(mx.cpu())
#
#print "====================results=================================="
#print speedup_sum
#print float(speedup_sum)/loop
#


##
#cv.finetune_codebooks()

#for k in cv.in_args:
#    print k
#print "==========="
#
#
#for k in cv.codebook_args:
#    print k

#train(prefix,epoch)