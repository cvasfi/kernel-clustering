import mxnet as mx
import numpy as np
from ResNet import resnet
import time
import copy
import logging


class converter(object):

    def __init__(self, in_prefix,in_epoch, batch_size, data_path = "dataset/cifar10_val.rec", shrink = 8):
        self.layers = []
        self.network = {}
        self.frozen_params=[]
        self.in_sym, self.in_args, self.in_auxs = mx.mod.module.load_checkpoint(in_prefix, in_epoch)
        self.prefix=in_prefix
        self.epoch = in_epoch
        self.batch_size = batch_size
        self.shrink=shrink

        self.val_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32), path_imgrec=data_path)
        self.input_shape = self.val_iter.provide_data[0][1]
        self.codebook_args = copy.deepcopy(self.in_args)

        self.process_symbol()


        auglist = mx.image.CreateAugmenter((3, 32, 32), resize=0, rand_mirror=True, hue=0.3, brightness=0.4,
                                           saturation=0.3, contrast=0.35, rand_crop=True, rand_gray=0.3)
        self.train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32),
                                        path_imgrec="dataset/cifar10_train.rec", aug_list=auglist)


    def extract_cluster_centers(self, data):
        shape = data.shape
        nclusters = shape[0] / self.shrink
        q_indices = np.zeros((shape[0], shape[1]))
        codewords = np.zeros((nclusters, shape[1], shape[2], shape[3]))

        for channel in range(shape[1]):
            cluster_centers, reconstructed_indices = np.unique(data[:, channel, :, :].asnumpy(), return_inverse=True,
                                                               axis=0)
            codewords[:, channel, :, :] = cluster_centers
            q_indices[:, channel] = reconstructed_indices

        return codewords, q_indices

    def get_onehot(self, data, nclusters, batch_size):
        index_mat = mx.nd.one_hot(mx.nd.array(data), depth=nclusters).reshape(0, -1)
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

                codebook_filter, indices = self.extract_cluster_centers(weight)
                lrshape = in_sym.get_internals()[layer + "_output"].infer_shape(data=in_shape)[1]
                codebook_flattened =   mx.nd.transpose(mx.nd.array(codebook_filter), axes=(1,0,2,3)).reshape((-1,1,0, 0)) # TODO: put this in codebook extraction function
                onehot_indices = self.get_onehot(indices,indices.shape[0]/self.shrink,self.batch_size)
                self.layers.append(layer)
                self.network[layer] = {}
                self.codebook_args[layer+"_weight"] = mx.nd.array(codebook_flattened)
                self.codebook_args[layer + "_indices"] = onehot_indices
                self.frozen_params.append(layer + "_indices")
                self.network[layer]["f_shape"] = codebook_filter.shape
                self.network[layer]["c_shape"] = codebook_flattened.shape
                self.network[layer]["i_shape"] = onehot_indices.shape
                self.network[layer]["out_shape"] = lrshape[0]
        print "symbol processed, clusters extracted."


    def convolve_codebook_light(self, data, name, num_filter = None, kernel=(3,3), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace = None):
        layer = self.network[name]
        fshape = layer["f_shape"]
        codebookshape = layer["c_shape"]
        indices_shape = layer["i_shape"]
        output_shape = layer["out_shape"]


        filters = mx.sym.Variable(name+"_weight", shape=codebookshape)
        indices = mx.sym.Variable(name+"_indices", shape=indices_shape)

        res = mx.sym.Convolution(data=data, weight=filters, num_group=fshape[1], num_filter=fshape[0] * fshape[1], stride = stride,
                                 no_bias=no_bias, kernel=kernel, workspace =workspace, pad=pad, name=name)
        res = res.reshape((0, 0, -1))  # flatten the image for matmul lookup

        res = mx.sym.batch_dot(lhs=indices, rhs=res, name=name+"_dot")

        res = res.reshape((0, 0, output_shape[2], output_shape[3]),name=name+"_reshape")

        return res

    def convert(self):
        depth = 20
        per_unit = [(depth - 2) / 6]
        filter_list = [16, 16, 32, 64]
        bottle_neck = False
        units = per_unit * 3
        self.converted_sym = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=10, data_type="cifar10",
                     bottle_neck=bottle_neck, custom_conv=self.convolve_codebook_light)

    def get_score(self,symbol_input, in_args, in_auxs):
        mod = mx.mod.Module(symbol=symbol_input, context=mx.cpu())
        mod.bind(for_training=False, data_shapes=self.val_iter.provide_data, label_shapes=self.val_iter.provide_label)
        mod.set_params(in_args, in_auxs)
        begin = time.time()

        score= mod.score(self.val_iter, ['acc'])

        duration = time.time() - begin
        return score, duration

    def evaluate_converted(self):
        return self.get_score(self.converted_sym, self.codebook_args, self.in_auxs)

    def compare_baseline(self):
        score1, time1 = self.get_score(self.in_sym,self.in_args,self.in_auxs)
        score2, time2 = self.evaluate_converted()

        print "\nOriginal network score: {}, time to complete: {}".format(score1,time1)
        print "\nClustered network score: {}, time to complete: {}".format(score2,time2)
        print ("\nCompressed {}x, Speedup: {}".format(self.shrink,float(time1)/time2))


    def finetune_codebooks(self):
        logging.getLogger().setLevel(logging.DEBUG)

        mod = mx.mod.Module(symbol=self.converted_sym, context=mx.gpu(),fixed_param_names=self.frozen_params)
        optimizer_params = {'learning_rate': 0.00005,
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
                epoch_end_callback=mx.callback.do_checkpoint(prefix+"_finetuned"),
                arg_params=self.codebook_args,
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
prefix="cnn_models/naive8x/resnet20_clustered_naive_8x_finetuned"
epoch=6
#cv = converter(prefix, epoch, batch_size=32, shrink=8)
#cv.convert()
##print cv.compare_baseline()
##
#cv.finetune_codebooks()

#for k in cv.in_args:
#    print k
#print "==========="
#
#
#for k in cv.codebook_args:
#    print k

train(prefix,epoch)