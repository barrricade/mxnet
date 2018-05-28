from mxnet import gluon,init,nd,autograd
from mxnet.gluon import loss as gloss,data as gdata,nn
import pandas as pd
import numpy as np
import gluonbook as gb
train_data = pd.read_csv("/home/hansome/workspace/data/train.csv")
test_data = pd.read_csv("/home/hansome/workspace/data/test.csv")
#pandas数据的操作，逗号前：为每一行，逗号后为从1列到最后1列
all_features = pd.concat((train_data.loc[:, 'MSSubClass':'SaleCondition'],test_data.loc[:, 'MSSubClass':'SaleCondition']))
#查看所有数值特征
numeric_features = all_features.dtypes[all_features.dtypes!="object"].index
#用均值填冲空值
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x:(x-x.mean())/(x.std()))
#对非数值特征进行转换
all_features = pd.get_dummies(all_features,dummy_na=True)
all_features = all_features.fillna(all_features.mean())
#行数
n_train = train_data.shape[0]
train_features = all_features[:n_train].as_matrix()
test_featurns = all_features[n_train:].as_matrix()
#SalePrice作为输出labels
train_labels = train_data.SalePrice.as_matrix()
train_features = nd.array(train_features)
train_labels = nd.array(train_labels)
train_labels.reshape((n_train,1))
test_features = nd.array(test_featurns)
#定义了损失函数
loss = gloss.L2Loss()
def get_rmse_log(net,train_features,train_labels):
    clipped_preds = nd.clip(net(train_features),1,float('inf'))
    return nd.sqrt(2*loss(clipped_preds.log(),
                          train_labels.log()).mean()).asnumpy()
#定义网络
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(23, activation='relu'))
    net.add(nn.Dense(1))
    net.initialize(init=init.Xavier())
    return net
#训练过程
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, verbose_epoch, learning_rate, weight_decay, batch_size):
    train_ls = []
    if test_features is not None:
        test_ls = []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了 Adam 优化算法。
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(1, num_epochs + 1):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            cur_train_l = get_rmse_log(net, train_features, train_labels)
        if epoch >= verbose_epoch:
            print("epoch %d, train loss: %f" % (epoch, cur_train_l))
        train_ls.append(cur_train_l)
        if test_features is not None:
            cur_test_l = get_rmse_log(net, test_features, test_labels)
            test_ls.append(cur_test_l)
    '''if test_features is not None:
        gb.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
                    range(1, num_epochs+1), test_ls, ['train', 'test'])
    else:
        gb.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss')'''
    if test_features is not None:
        return cur_train_l, cur_test_l
    else:
        return cur_train_l
#k折交叉模型用来调参
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay, batch_size):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_l_sum = 0.0
    test_l_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]
        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_l, test_l = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, verbose_epoch, learning_rate, weight_decay, batch_size)
        train_l_sum += train_l
        print("test loss: %f" % test_l)
        test_l_sum += test_l
    return train_l_sum / k, test_l_sum / k
#预测房价
def train_and_pred(num_epochs, verbose_epoch, train_features, test_feature,
                   train_labels, test_data, lr, weight_decay, batch_size):
    net = get_net()
    train(net, train_features, train_labels, None, None, num_epochs,
          verbose_epoch, lr, weight_decay, batch_size)
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission3.csv', index=False)
k = 5
num_epochs = 100
verbose_epoch = num_epochs - 2
lr = 5
weight_decay = 0
batch_size = 64
net  = get_net()
train_and_pred(num_epochs, verbose_epoch, train_features, test_features,
               train_labels, test_data, lr, weight_decay, batch_size)
'''epoch 98, train loss: 0.163138
epoch 99, train loss: 0.162845
epoch 100, train loss: 0.162497'''