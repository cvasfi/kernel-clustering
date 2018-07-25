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
parser.add_argument('--shrink', default="2", type=int,help='val folder')
parser.add_argument('--lr', default="0.00001", type=float,help='val folder')


args = parser.parse_args()


cl = ClusterNet(args.prefix, args.epoch, args.batch_size, process=False, shrink = args.shrink, dataset="imagenet", traindir=args.trainrec, valdir=args.valrec, arch='vgg', lr=args.lr)
cl.finetune_codebooks()