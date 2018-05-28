import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon,autograd,nd,init
from mxnet.gluon import loss as gloss,nn
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)
def accuracy(y_, y):
    return (y_.argmax(axis=1) == y).mean().asscalar()
batch_size = 256
train_iter,test_iter = gb.load_data_fashion_mnist(batch_size)
net = nn.Sequential()
net.add(nn.Flatten())
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
#损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
#定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 5
#训练
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        train_l_sum = 0
        train_acc_sum = 0
        with autograd.record():
            y_ = net(X)
            l = loss(y_, y)
        l.backward()
        trainer.step(batch_size)    
        train_l_sum += l.mean().asscalar()
        train_acc_sum += accuracy(y_, y) 
    test_acc = evaluate_accuracy(test_iter, net) 
    print(train_l_sum/len(train_iter),train_acc_sum/len(train_iter),test_acc)
