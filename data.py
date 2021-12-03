import numpy as np

def generate_data(degree, num_datapoints, noise, minimum, maximum):
    coef = np.random.random(degree+1)
    coef /= np.linalg.norm(coef)
    x = np.random.random(num_datapoints)*(maximum-minimum)+minimum
    y = np.power(x.reshape(-1,1), np.repeat(np.arange(degree+1).reshape(1,-1), num_datapoints, axis=0))
    y = np.dot(y, coef.reshape(-1,1))
    return x.reshape(-1, 1), (y+np.random.randn(num_datapoints,1)*noise)


# print(generate_fun(1,5,0,0,1))
