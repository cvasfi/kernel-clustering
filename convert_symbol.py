import mxnet as mx
import numpy as np
from ResNet import resnet

class converter(object):

    def __init__(self, in_prefix,in_epoch, batch_size, data_path = "dataset/cifar10_val.rec", shrink = 8):
        self.layers = []
        self.network = {}
        self.in_sym, self.in_args, self.in_auxs = mx.mod.module.load_checkpoint(in_prefix, in_epoch)
        self.prefix=prefix
        self.batch_size = batch_size
        self.shrink=shrink

        self.val_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32), path_imgrec=data_path)
        self.input_shape = self.val_iter.provide_data[0][1]
        self.codebook_args = self.in_args

        self.process_symbol()


        #auglist = mx.image.CreateAugmenter((3, 32, 32), resize=0, rand_mirror=True, hue=0.3, brightness=0.4,
        #                                   saturation=0.3, contrast=0.35, rand_crop=True, rand_gray=0.3)
        #train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 32, 32),
        #                                path_imgrec="dataset/cifar10_train.rec", aug_list=auglist)


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
            if "conv" in element and "weight" in element:  # maybe tune for different networks
                if "conv0" in element:  # hack to exclude first layer
                    continue
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
                self.network[layer]["f_shape"] = codebook_filter.shape
                self.network[layer]["c_shape"] = codebook_flattened.shape
                self.network[layer]["i_shape"] = onehot_indices.shape
                self.network[layer]["out_shape"] = lrshape[0]
        print "symbol processed, clusters extracted."

    def convolve_codebook(self, data, name, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace = None):
        layer=self.network[name]
        indices=layer["indices"]
        codebookshape = layer["c_shape"]
        output_shape = layer["out_shape"]

        filters = mx.sym.Variable(name+"_weight", shape=codebookshape)
        fshape = codebookshape  # 4,16,3,3
        index_shape = indices.shape

        filters = mx.sym.transpose(filters, axes=(1, 0, 2, 3)).reshape((-1, 1, 0, 0))  # TODO: transpose is unnecessary!!
        res = mx.sym.Convolution(data=data, weight=filters, num_group=fshape[1], num_filter=fshape[0] * fshape[1],
                                  kernel=kernel, stride=stride, pad=pad, no_bias=no_bias, workspace=workspace)
        res = res.expand_dims(1)
        res = res.reshape((0, fshape[1], fshape[0], 0, 0))
        res = mx.sym.transpose(res, axes=(0, 2, 1, 3, 4))  # lookup table

        # hacky because multi-dim indexing isn't allowed
        res = mx.sym.reshape(data=res, shape=(-1, 0), reverse=1)  # (sample*nclusters*channel*W,H)
        # now looking up the results

        # print res[0,1,0] #7, 4, 16 ,30, 30
        #print index_shape  # 7,4,16,30,30
        lres = []
        # TODO: find a way to implement with less loops
        for sample in range(output_shape[0]):
            filterwise_list = []
            for fltr in range(index_shape[0]):
                channelwise_list = []
                for ch in range(index_shape[1]):
                    ## (((sample*4+cluster)*channels)*channel)*width
                    slice_begin = (int(((sample * fshape[0] + indices[fltr, ch]) * fshape[1] + ch) * output_shape[2]), 0)
                    slice_end = (int(slice_begin[0] + output_shape[2]), int(output_shape[3]))

                    # channelwise_list.append(res[sample][indices[fltr,ch]][ch][0])
                    channelwise_list.append(mx.sym.slice(data=res, begin=slice_begin, end=slice_end))

                filterwise_list.append(mx.sym.sum(mx.sym.stack(*channelwise_list), axis=0))
            lres.append(mx.sym.stack(*filterwise_list))
        lres = mx.sym.stack(*lres)

        return lres

    def convolve_codebook_light(self, data, name, num_filter = None, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace = None):
        layer = self.network[name]
        fshape = layer["f_shape"]
        codebookshape = layer["c_shape"]
        indices_shape = layer["i_shape"]
        output_shape = layer["out_shape"]


        filters = mx.sym.Variable(name+"_weight", shape=codebookshape)
        indices = mx.sym.Variable(name+"_indices", shape=indices_shape)
        # fshape  = codebookshape #4,16,3,3

        # filters = mx.sym.transpose(filters, axes=(1,0,2,3)).reshape((-1,1,0, 0)) #TODO: transpose is unnecessary!!
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
        mod = mx.mod.Module(symbol=symbol_input, context=mx.gpu())
        mod.bind(for_training=False, data_shapes=self.val_iter.provide_data, label_shapes=self.val_iter.provide_label)
        mod.set_params(in_args, in_auxs)
        return mod.score(self.val_iter, ['acc'])

    def predict_converted(self):
        return self.get_score(self.converted_sym, self.codebook_args, self.in_auxs)

    #def compare_baseline(self):


#
prefix="cnn_models/resnet20_clustered"
epoch=0
cv = converter(prefix, epoch, batch_size=8, shrink=4)
cv.convert()
print cv.predict_converted()

#for k in cv.in_args:
#    print k
#print "==========="
#
#
#for k in cv.codebook_args:
#    print k