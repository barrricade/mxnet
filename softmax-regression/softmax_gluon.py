import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
#初始化模型
net = nn.Sequential()#代表全连接层
net.add(nn.Flatten())#图片压成向量
net.add(nn.Dense(10))#增加10个节点的输出层
net.initialize(init.Normal(sigma=0.01))
#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
#定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 5
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             None, trainer)