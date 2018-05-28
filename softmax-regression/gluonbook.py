from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels +=nd.random.normal(scale=0.01,shape=labels.shape)
def plt(features,labels):
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    plt.show()
batch_size = 10
def data_iter(batch_size, num_examples, features, labels):
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
def linreg(X,w,b):
    return nd.dot(X,w)+b
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param-lr*param.grad/batch_size
lr = 0.03
batch_size = 10
num_epochs = 3
net = linreg
loss = squared_loss
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w, b]
for param in params:
    param.attach_grad()
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter(batch_size, num_examples, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
def train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, trainer=None):
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"% (epoch, train_l_sum / len(train_iter),train_acc_sum / len(train_iter), test_acc))
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y).mean().asscalar()
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)

def transform(feature, label):
    return feature.astype('float32') / 255, label.astype('float32')
def load_data_fashion_mnist(batch_size):
    mnist_train = gdata.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gdata.vision.FashionMNIST(train=False, transform=transform)
    train_iter = gdata.DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = gdata.DataLoader(mnist_test, batch_size, shuffle=False)
    return train_iter,test_iter