import mxnet as mx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

prefix="cnn_models/resnet20"
epoch=124

sym, args, auxs = mx.mod.module.load_checkpoint(prefix, epoch)

batch_size=32
val_iter=mx.image.ImageIter(batch_size=batch_size,data_shape=(3,32,32),path_imgrec="dataset/cifar10_val.rec")
testarray=val_iter.next().data

test_iter=mx.io.NDArrayIter(testarray,batch_size=batch_size)

speedups=[]


def get_tensor(symbol_input):
    mod=mx.mod.Module(symbol=symbol_input, context=mx.cpu(),label_names=None)
    mod.bind(for_training=False, data_shapes=test_iter.provide_data)
    mod.set_params(args, auxs)
    return mod.predict(eval_data=test_iter)


def get_layer_sqr_error(in_layer,out_layer, layer_weights, shrink):
    sym_in = sym.get_internals()[in_layer]
    sym_out_original = sym.get_internals()[out_layer]

    tensor_in = get_tensor(sym_in)
    tensor_out_original = get_tensor(sym_out_original)
    num_filter=tensor_out_original.shape[1]
    stride=tensor_in.shape[3]//tensor_out_original.shape[3]


    filters = args[layer_weights]

    filters_quantized_reshaped=get_quantized(filters,16)

    clustered_result = mx.ndarray.Convolution(data=tensor_in, weight=filters_quantized_reshaped, num_filter=num_filter,
                                              kernel=(3, 3), stride=(stride, stride), pad=(1, 1)
                                              , no_bias=True, name="conv0")

    return np.square(tensor_out_original.asnumpy() - clustered_result.asnumpy()).mean()


#def get_quantized(filters, shrink=16):
#    shape=filters.shape
#    n_clusters=shape[0]*shape[1] / shrink
#
#    filters_shaped=filters.reshape((shape[0] * shape[1], shape[2] * shape[3]))
#    estimator = KMeans(n_clusters=n_clusters)
#    estimator.fit(filters_shaped.asnumpy())
#
#    filter_kmean_indexes = estimator.predict(X=filters_shaped.asnumpy())
#    filters_quantized = np.array([estimator.cluster_centers_[idx] for idx in filter_kmean_indexes])
#    filters_quantized = mx.nd.array(filters_quantized)
#
#    return filters_quantized.reshape(shape)

def get_quantized(filters, shrink=16):
    shape=filters.shape
    n_clusters=shape[0]*shape[1]/16

    filters_shaped=filters.reshape((shape[0] * shape[1], shape[2] * shape[3]))
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(filters_shaped.asnumpy())

    filter_kmean_indexes = estimator.predict(X=filters_shaped.asnumpy())
    filters_quantized = np.array([estimator.cluster_centers_[idx] for idx in filter_kmean_indexes])
    filters_quantized = mx.nd.array(filters_quantized)



    fq= filters_quantized.reshape(shape)

    sum=0

    for channel in range(shape[1]):
        filters_in_channel = fq[:,channel,:,:]
        nclusters_channel = np.unique(filters_in_channel.asnumpy(),axis=0)
        sum+=nclusters_channel.shape[0]
        #print nclusters_channel.shape[0]


    speedups.append(float(shape[0]*shape[1])/sum)

    return fq

def get_channelwise_clustered(filters, shrink):
    shape = filters.shape
    print shape
    result=np.zeros(shape)
    n_clusters = int(shape[0] / shrink)
    print filters[:,0,:,:].shape
    print n_clusters

    for channel_idx in range(shape[1]):

        channelwise_filters = filters[:,channel_idx,:,:]
        cw_shape=channelwise_filters.shape

        channelwise_filters_shaped = channelwise_filters.reshape((cw_shape[0], cw_shape[1] * cw_shape[2]))
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit(channelwise_filters_shaped.asnumpy())

        cw_filter_kmean_indexes = estimator.predict(X=channelwise_filters_shaped.asnumpy())
        cw_filters_quantized = np.array([estimator.cluster_centers_[idx] for idx in cw_filter_kmean_indexes])

        result[:,channel_idx,:,:]=cw_filters_quantized.reshape(cw_shape)



    return mx.nd.array(result)


def get_score(in_sym,in_args,in_auxs):
    mod = mx.mod.Module(symbol=in_sym, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
    mod.set_params(in_args, in_auxs)
    return mod.score(val_iter, [mx.metric.Accuracy()])

def naive_clusternet():
    for l in sym.get_internals().list_outputs():
        if "weight" in l and "conv" in l:
            if 'conv0' in l:
                print l
                pass
            else:
                #args[l] = get_quantized(args[l], 8)
                args[l] = get_channelwise_clustered(args[l], 2)

    print get_score(sym,args,auxs)


def reshape_insert_filters(filters,fshapes):

    indexer=0
    print "shape of all the kernels: {}".format(filters.shape)
    for fshape in fshapes:
        name=fshape[0]
        shape = fshape[1]
        nkernels=shape[0]*shape[1]
        print shape
        print("for layer {} with shape {}, setting the index to {}:{}".format(name, shape, indexer, indexer+nkernels))
        filter=filters[indexer:indexer+nkernels,:]
        filter=filter.reshape(shape)
        indexer=indexer+nkernels
        args[name]=filter

        sum = 0

        for channel in range(shape[1]):
            filters_in_channel = filter[:, channel, :, :]
            nclusters_channel = np.unique(filters_in_channel.asnumpy(), axis=0)
            sum += nclusters_channel.shape[0]
            # print nclusters_channel.shape[0]

        speedups.append(float(shape[0] * shape[1]) / sum)


def get_speedup(fshapes):
    for fshape in fshapes:
        name=fshape[0]
        shape = fshape[1]




def global_clusternet(shrink=32):    #TODO: use a generalized cluster function

    filters = []
    fshapes=[]

    for l in sym.get_internals().list_outputs():
        if "weight" in l and "conv" in l:
            shape=args[l].shape
            filter=args[l].reshape((shape[0] * shape[1], shape[2] * shape[3]))
            filters.append(filter)
            fshapes.append((l,shape))



    filters=mx.nd.concatenate(filters)

    estimator = KMeans(n_clusters=filters.shape[0]/shrink)
    estimator.fit(filters.asnumpy())

    filter_kmean_indexes = estimator.predict(X=filters.asnumpy())
    filters_quantized = np.array([estimator.cluster_centers_[idx] for idx in filter_kmean_indexes])
    filters_quantized = mx.nd.array(filters_quantized)

    reshape_insert_filters(filters_quantized,fshapes)

    print get_score(sym,args,auxs)


#def iterative_finetuned_clusternet():
#    for l in sym.get_internals().list_outputs():
#        if "weight" in l and "conv" in l:
#            if 'conv0' in l:
#                print l
#                pass
#            else:
#                args[l] = get_quantized(args[l], 8)
#
#    print get_score(sym,args,auxs)


#global_clusternet(32)
#print get_layer_sqr_error('stage3_unit1_relu1_output','stage3_unit1_conv1_output','stage3_unit1_conv1_weight',16)

naive_clusternet()

#print get_layer_sqr_error('bn_data_output','conv0_output','conv0_weight',24)

print speedups
#
print np.mean(np.array(speedups))