
import mxnet as mx
import numpy as np

def get_symbol(num_classes, customconv=mx.sym.Convolution, dtype='float32', **kwargs):
    input_data = mx.sym.Variable(name="data")
    if dtype == 'float16':
        input_data = mx.sym.Cast(data=input_data, dtype=np.float16)
    # stage 1
    conv1 = customconv(name='conv1',
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu", name='relu1')
    lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=1, nsize=5, name='norm1')
    pool1 = mx.sym.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2), name="pool1", pad=(0,0))
    # stage 2
    conv2 = customconv(name='conv2',
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu", name='relu2')
    lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5, name='norm2')
    pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool2")
    # stage 3
    conv3 = customconv(name='conv3',
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu", name='relu3')
    conv4 = customconv(name='conv4',
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.sym.Activation(data=conv4, act_type="relu", name='relu4')
    conv5 = customconv(name='conv5',
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.sym.Activation(data=conv5, act_type="relu", name='relu5')
    pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool5")
    # stage 4
    flatten = mx.sym.Flatten(data=pool3, name="flatten_0")
    fc1 = mx.sym.FullyConnected(name='fc6', data=flatten, num_hidden=4096)
    relu6 = mx.sym.Activation(data=fc1, act_type="relu", name='relu6')
    dropout1 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    # stage 5
    fc2 = mx.sym.FullyConnected(name='fc7', data=dropout1, num_hidden=4096)
    relu7 = mx.sym.Activation(data=fc2, act_type="relu", name='relu7')
    dropout2 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
    # stage 6
    fc3 = mx.sym.FullyConnected(name='fc8', data=dropout2, num_hidden=num_classes)
    if dtype == 'float16':
        fc3 = mx.sym.Cast(data=fc3, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return softmax



def train():
    trainsize = 1
    testsize = 1
    numclasses = 1000
    batchsize=1
    ctx=mx.gpu()
    prefix="alexnet_test"

    train_data = mx.nd.random_uniform(shape=(trainsize, 3, 224, 224))
    train_labels = mx.nd.one_hot(depth=numclasses, indices= mx.nd.array(np.random.choice(numclasses,trainsize)))

    #test_data  = mx.nd.random_uniform(shape=(testsize, 3, 224, 224))
    #test_labels = mx.nd.one_hot(depth=numclasses, indices= mx.nd.array(np.random.choice(numclasses,testsize)))

    train_iter = mx.io.NDArrayIter(train_data, train_labels, batchsize)
    #test_iter = mx.io.NDArrayIter(test_data, test_labels, batchsize)

    sym=get_symbol(numclasses)
    args = None
    auxs = None

    mod = mx.mod.Module(symbol=sym, context=ctx)


    optimizer_params = {'learning_rate': 0.001,
                       'momentum': 0.9,
                       'wd': 0.0005,
                       'clip_gradient': None,
                       'rescale_grad': 1.0}

    mod.fit(train_iter,
                    #eval_data=test_iter,
                    optimizer='sgd',
                    optimizer_params=optimizer_params,
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batchsize, 100),
                    epoch_end_callback=mx.callback.do_checkpoint(prefix),
                    arg_params=args,
                    aux_params=auxs,
                    begin_epoch=0,
                    num_epoch=1)



train()