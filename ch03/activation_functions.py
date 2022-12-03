import numpy as np
import matplotlib.pyplot as plt


def step(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


# cannot handle overflow (cf. p.69)
def old_softmax(a):
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def main():
    x = np.arange(-5, 5, 0.1)
    y1 = step(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim([-0.1, 1.1])
    plt.legend(["step", "sigmoid", "relu"])
    plt.show()


if __name__ == "__main__":
    main()
