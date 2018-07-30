import mxnet as mx
import time
import numpy as np


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--valrec', default="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec", type=str,help='val folder')

parser.add_argument('--prefix_orig', default="/mnt/data/vgg/vgg16", type=str,help='val folder')
parser.add_argument('--epoch_orig', default="0", type=int,help='val folder')

parser.add_argument('--prefix', default="/mnt/data/vgg/vgg16", type=str,help='val folder')
parser.add_argument('--epoch', default="0", type=int,help='val folder')

parser.add_argument('--batch_size', default="2", type=int,help='val folder')
parser.add_argument('--ctx', default="gpu", type=str,help='val folder')


args = parser.parse_args()

ctx = mx.gpu() if args.ctx=="gpu" else mx.cpu()


tempiter = mx.io.ImageRecordIter(path_imgrec=args.valrec, data_name="data", label_name="softmax_label",
                batch_size=1000, data_shape=(3, 224, 224), shuffle=True)

firstbatch = tempiter.next()
nd_data = firstbatch.data
nd_label= firstbatch.label
val_iter = mx.io.NDArrayIter(nd_data,  label=nd_label,batch_size=args.batch_size)


def get_forward(symbol_input, in_args, in_auxs, val_iter,  evalctx=mx.cpu(), label_names=["softmax_label"]):
    mod = mx.mod.Module(symbol=symbol_input, context=evalctx, label_names=label_names)
    mod.bind(for_training=False, data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
    mod.set_params(in_args, in_auxs)
    begin = time.time()

    score = mod.predict(val_iter).asnumpy()

    duration = time.time() - begin
    return score, duration

sym_orig, args_orig, auxs_orig = mx.mod.module.load_checkpoint(args.prefix_orig, args.epoch_orig)

sym_q, args_q, auxs_q = mx.mod.module.load_checkpoint(args.prefix, args.epoch)


tempiter = mx.io.ImageRecordIter(path_imgrec=args.valrec, data_name="data", label_name="softmax_label",
                batch_size=1000, data_shape=(3, 224, 224), shuffle=True)

firstbatch = tempiter.next()
nd_data = firstbatch.data
nd_label= firstbatch.label
val_iter = mx.io.NDArrayIter(nd_data,  label=nd_label,batch_size=args.batch_size)


score1, time1 = get_forward(sym_orig, args_orig, auxs_orig, val_iter, ctx)
print "\nOriginal network score: {}, time to complete: {}".format(score1.shape, time1)


score2, time2 = get_forward(sym_q, args_q, auxs_q, val_iter, ctx)
print "\nClustered network score: {}, time to complete: {}".format(score2.shape, time2)

print "speedup: {}".format(time1/time2)

print np.mean(np.square(score2-score1))

