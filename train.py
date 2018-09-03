import mxnet as mx
import logging

import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--trainrec', default="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec", type=str,help='train folder')
parser.add_argument('--valrec', default="/mnt/data/dataset_stuff/imagenetval/imagenet1k-val.rec", type=str,help='val folder')
parser.add_argument('--prefix', default="/mnt/data/vgg/vgg16", type=str,help='val folder')
parser.add_argument('--epoch', default="0", type=int,help='val folder')
parser.add_argument('--batch_size', default="1", type=int,help='val folder')
parser.add_argument('--lr', default="0.00001", type=float,help='val folder')
parser.add_argument('--ctx', default="gpu", type=str,help='val folder')
parser.add_argument('--resume', default=None, type=bool, help='')



def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]

    lr = learning_rate
    epoch_size = num_example // batch_size
    for s in iter_refactor:
        if begin_epoch >= s:
            lr *= lr_refactor_ratio
    if lr != learning_rate:
        logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
    if not steps:
        return (lr, None)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
    return (lr, lr_scheduler)


def train(traindir, valdir, prefix, epoch=10,  resume=True, batch_size= 32, lr = 0.001, ctx=mx.cpu()):





    auglist = mx.image.CreateAugmenter((3, 224, 224), resize=0, rand_mirror=True, hue=0.4, brightness=0.4,
                                       saturation=0.4, contrast=0.4, rand_crop=True, rand_gray=0.4, rand_resize=True,
                                       pca_noise=0.2)
    train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3, 224, 224),
                                         path_imgrec=traindir, aug_list=auglist)

    val_iter = mx.io.ImageRecordIter(path_imgrec=valdir, data_name="data", label_name="softmax_label",
                                          batch_size=batch_size, data_shape=(3, 224, 224))



    if resume:
        sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)
    else:
        sym =  mx.sym.load(prefix)
        args = None
        auxs = None

    mod = mx.mod.Module(symbol=sym, context=ctx)

    learning_rate, lr_scheduler = get_lr_scheduler(lr, '80, 160',
                                                   0.1, 48638, 128, epoch)

    optimizer_params = {'learning_rate': learning_rate,
                       'momentum': 0.9,
                       'wd': 0.0005,
                        'lr_scheduler':lr_scheduler,
                       'clip_gradient': None,
                       'rescale_grad': 1.0}

    mod.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params=optimizer_params,
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    epoch_end_callback=mx.callback.do_checkpoint(prefix),
                    arg_params=args,
                    aux_params=auxs,
                    begin_epoch=epoch,
                    num_epoch=300)


if __name__ =='__main__':
    args = parser.parse_args()
    ctx = mx.gpu() if args.ctx == "gpu" else mx.cpu()

    train(args.trainrec, args.valrec, args.prefix, args.epoch, args.resume, args.batch_size, args.lr, ctx)