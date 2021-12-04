import numpy as np

class PolyDataGen:
   def __init__(self, degree):
       self.degree = degree
       coef = np.random.random(degree+1)
       coef /= np.linalg.norm(coef)
       self.coef = coef

   def generate_data(self, num_datapoints, noise, minimum, maximum):
       x = np.random.random(num_datapoints)*(maximum-minimum)+minimum
       y = np.power(x.reshape(-1,1), np.repeat(np.arange(self.degree+1).reshape(1,-1), num_datapoints, axis=0))
       y = np.dot(y, self.coef.reshape(-1,1))
       return x.reshape(-1, 1), (y+np.random.randn(num_datapoints,1)*noise)

   def add_noise(self, x, noise):
       '''
       x <- (D,) or (D, 1)
       noise <- std
       '''   
       y = np.power(x.reshape(-1,1), np.repeat(np.arange(self.degree+1).reshape(1,-1), num_datapoints, axis=0))
       y = np.dot(y, self.coef.reshape(-1,1))
       return x.reshape(-1, 1), (y+np.random.randn(num_datapoints,1)*noise)

class TrigDataGen:
   def __init__(self):
       self.phi = np.random.random()*2*np.pi
       self.a = np.random.random()*10

   def generate_data(self, num_datapoints, noise, minimum, maximum):
       x = np.random.random(num_datapoints)*(maximum-minimum)+minimum
    #    y = 1/x
       y = x
    #    y = (np.sin(self.a * x + self.phi))/x
       y = y.reshape((-1,1))
       return x.reshape(-1, 1), (y+np.random.randn(num_datapoints,1)*noise)


class InversePolyDataGen:
   def __init__(self):
       self.phi = np.random.random()*2*np.pi
       self.a = np.random.random()*10

   def generate_data(self, num_datapoints, noise, minimum, maximum):
       x = np.random.random(num_datapoints)*(maximum-minimum)+minimum
       y = (1/x+0.001)
       y = y.reshape((-1,1))
       return x.reshape(-1, 1), (y+np.random.randn(num_datapoints,1)*noise)



