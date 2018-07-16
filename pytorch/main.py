from vgg16_clusternet import vgg16_clusternet
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--traindir', default="/mnt/data/dataset_stuff/imagenet_dummy", type=str,help='train folder')
parser.add_argument('--valdir', default="/mnt/data/dataset_stuff/imagenet_dummy", type=str,help='val folder')
parser.add_argument('--weights', default="/home/tapir/Documents/Thesis/vgg16_pytoarch_weights/vgg16-397923af.pth", type=str,help='val folder')
parser.add_argument('--prefix', default="test", type=str,help='val folder')

args = parser.parse_args()



cn=vgg16_clusternet(weights=args.weights, shrink=4)


cn.train(traindir=args.traindir, valdir=args.valdir,batch_size=1,save_prefix=args.prefix)