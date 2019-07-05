import numpy as np

def step(x):
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    alpha = 0.01 #傾き
    return np.maximum(alpha * x, x)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__=="__main__":
    import matplotlib.pylab as plt
    x = np.arange(-5.0, 5.0, 0.1)
    y = leaky_relu(x) #デモ(関数を書き変えて選択)
    plt.plot(x, y)
    plt.ylim(-5.1, 5.1)
    plt.show()