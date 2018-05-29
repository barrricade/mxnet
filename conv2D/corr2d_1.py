from mxnet import nd,init,autograd,gluon
from mxnet.gluon import loss as gloss,data as gdata,nn
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y
x = nd.ones((6,8))
x[:,2:6] = 0
k = nd.array([[1,-1],[2,2]])
y = corr2d(x,k)
print(y.shape)
#初始化卷积层,输出通道为1，核数组形状是（1,2）的二维卷积层
conv2d = nn.Conv2D(1,kernel_size=(2,2))
conv2d.initialize()
#二维卷积层使用四维输出，格式为（批量大小，通道数，高，宽），这里的批量大小和通道为1
x = x.reshape((1,1,6,8))
y = y.reshape((1,1,5,7))

for i in range(10):
    with autograd.record():
        y_ = conv2d(x)
        loss = (y_-y)**2
        if i%2 == 1:
            print("batch%d,loss%.3f"%(i,loss.sum().asscalar()))
    loss.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
print(conv2d.weight.data())