import sys
sys.path.append('/home/hansome/workspace/mxnet/house_prise/gluonbook.py')
import gluonbook as gb
from mxnet import autograd,nd,init,gluon
from mxnet.gluon import loss as gloss,data as gdata,nn,utils as gutils
import mxnet as mx
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)

def accuracy(y_, y):
    return (y_.argmax(axis=1) == y).mean().asscalar()
def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
lr = 0.5
batch_size=256
net.initialize(force_reinit=True,init=init.Xavier())
train_iter,test_iter = gb.load_data_fashion_mnist(batch_size)
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : lr})
loss = gloss.SoftmaxCrossEntropyLoss()
num_epochs = 5
ctx = try_gpu()
net.initialize(ctx=ctx)
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        '''train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        losses = []'''
        with autograd.record():
            for x,y in train_data:
                y_hat=net(x)
                l = loss(y_hat,y)
            l.backward()
        print(net[0].weight.data())
train(train_iter, test_iter, net, loss,
            trainer, ctx, num_epochs=5)