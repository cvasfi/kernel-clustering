from clusternet_propogated_lookup import ClusterNet
import vgg
import mxnet as mx
import time



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


def get_score(symbol_input, in_args, in_auxs, val_iter, evalctx=mx.cpu()):
    mod = mx.mod.Module(symbol=symbol_input, context=evalctx)
    mod.bind(for_training=False, data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
    mod.set_params(in_args, in_auxs)
    begin = time.time()

    score = mod.score(val_iter, ['acc'])

    duration = time.time() - begin
    return score, duration

_, args_orig, auxs_orig = mx.mod.module.load_checkpoint(args.prefix_orig, args.epoch_orig)
sym_orig = vgg.get_symbol(1000,16)

sym, args_q, auxs = mx.mod.module.load_checkpoint(args.prefix, args.epoch)


val_iter = mx.io.ImageRecordIter(path_imgrec=args.valrec, data_name="data", label_name="softmax_label",
                batch_size=args.batch_size, data_shape=(3, 224, 224))

score1, time1 = get_score(sym, args_q, auxs, val_iter, ctx)
print "\nClustered network score: {}, time to complete: {}".format(score1, time1)

score2, time2 = get_score(sym_orig, args_orig, auxs_orig, val_iter, ctx)
print "\nOriginal network score: {}, time to complete: {}".format(score2, time2)

speedup = float(time2) / time1
print ("\nCompressed 2x, Speedup: {}".format( speedup))