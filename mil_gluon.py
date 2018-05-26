import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon,autograd,nd,init
from mxnet.gluon import loss as gloss,nn
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
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)        
    print("epoch %d, loss: %f"
          % (epoch, loss(net(X), y).mean().asnumpy()))
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)