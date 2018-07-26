from clusternet_propogated_lookup import ClusterNet
import vgg
import mxnet as mx
import time



import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--trainrec', default="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec", type=str,help='train folder')
parser.add_argument('--valrec', default="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec", type=str,help='val folder')
parser.add_argument('--prefix', default="/mnt/data/vgg/vgg16", type=str,help='val folder')
parser.add_argument('--epoch', default="0", type=int,help='val folder')
parser.add_argument('--batch_size', default="1", type=int,help='val folder')
parser.add_argument('--lr', default="0.00001", type=float,help='val folder')


args = parser.parse_args()

#prefix= "cnn_models/resnet20"
#epoch=124
#batch_size = 32

#cl = ClusterNet(prefix, epoch, batch_size, process=True, shrink = 2, dataset="cifar10", imagenetpath=None, arch='resnet')
#cl.compare_baseline()

imagenetpath="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec"

#val_iter=mx.io.ImageRecordIter(
#    path_imgrec=imagenetpath, data_name="data", label_name="softmax_label",
#    batch_size=8, data_shape=(3, 224, 224))



#sym=vgg.get_symbol(1000, 16)
#_, args, auxs=mx.model.load_checkpoint("/mnt/data/vgg/vgg16", 0)

#mod = mx.mod.Module(symbol=sym, context=mx.gpu())
#mod.bind(for_training=False, data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
#mod.set_params(args, auxs)

#begin = time.time()
#score= mod.score(val_iter, ['acc'])
#print time.time()-begin

#print score



prefix="/mnt/data/vgg/vgg16"
epoch=0

cl = ClusterNet(args.prefix, args.epoch, args.batch_size, process=True, shrink = 2, dataset="imagenet", traindir=args.trainrec, valdir=args.valrec, arch='vgg', lr= args.lr)
cl.finetune_codebooks()