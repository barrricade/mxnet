from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,params,lr,trainer):
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_ = net(X)
                l = loss(y_, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()#返回一个标量
            train_acc_sum += accuracy(y_, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"% (epoch, train_l_sum / len(train_iter),
        train_acc_sum / len(train_iter), test_acc))
def accuracy(y_, y):
    return (y_.argmax(axis=1) == y).mean().asscalar()
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