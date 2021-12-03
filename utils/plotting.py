import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, label):
    '''
    Scatter Train/Test Datapoints
    '''
    plt.scatter(data[0].reshape((-1,)), data[1].reshape((-1,)), label=label)
    plt.legend()

def plot_fn(f, minimum, maximum, label):
    N_POINTS = 500
    x = np.linspace(minimum, maximum, N_POINTS)
    # plt.plot(x, [f(i) for i in x], linestyle="--", label=label)
    plt.plot(x, f(x).reshape((-1,)), linestyle="--", label=label)
    plt.legend()
