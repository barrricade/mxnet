from mxnet import gluon,init,nd,autograd
from mxnet.gluon import loss as gloss,data as gdata,nn
import pandas as pd
import numpy as np
import gluonbook as gb
train_data = pd.read_csv("/home/hansome/workspace/data/train.csv")
test_data = pd.read_csv("/home/hansome/workspace/data/test.csv")
all_features = pd.concat((train_data.loc[:, 'MSSubClass':'SaleCondition'],test_data.loc[:, 'MSSubClass':'SaleCondition']))
print(train_data.loc[0,'MSSubClass'])