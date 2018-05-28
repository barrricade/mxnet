from mxnet import gluon,nd,init,autograd
from mxnet.gluon import data as gdata,loss as gloss,nn
n_train = 20
n_test = 100
num_inputs = 200
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

features = nd.random.normal(shape=(n_train+n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

num_epochs = 10
learning_rate = 0.003
batch_size = 1
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)
loss = gloss.L2Loss()

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=1))
#对权重做L2范数正则化，权重衰减
train_w = gluon.Trainer(net.collect_params('.*weight'),'sgd',{'learning_rate':learning_rate,'wd':0})
#不对偏差参数做L2范数正则化
traun_b = gluon.Trainer(net.collect_params('.*bias'),'sgd',{'learning_rate':learning_rate})
for i in range(num_epochs):
    for x,y in range(train_iter):
        with autograd.record():
            l = loss(net(x),y)
        l.backward()
        train_w.step(batch_size)
        train_b.step(batch_size)
print(loss(net(features,labels)).mean().asscalar())
    