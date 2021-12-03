import numpy as np

def linear_solve(A, b):
  print(f"A.shape: {A.shape}, b.shape: {b.shape}")
  AAT = A@(A.T)
  print(f"AAT:{AAT.shape}")
  # w = np.linalg.solve(AAT, b)
  w = np.linalg.lstsq(AAT, b)[0]
  print(f"w:{w.shape}")
  x = (A.T)@w
  return x


class PolyModel:
  '''
  H(n) : h(x): R-> R, h(x)= a_0 + a_1*x + a_2*x^2 + ... + a_{n-1}*x^n-1 
  '''

  def __init__(self, n):
      self.n = n
      self.a = None
  
  def generate_features(self, X):
      X = X.flatten()
      arrays = [X**i for i in range(self.n)]
      A = np.stack(arrays, axis = 1)
      return A

  def fit(self, X, Y, refit=False):
      '''
      X: (d,1)
      Y: (d,1)
      '''
      if self.a and (not refit):
         raise ValueError("Re-Fitting")
      A = self.generate_features(X)
      self.a = linear_solve(A, Y)
      return self

  def predict(self, X, A=None):
      if (not A):
        A = self.generate_features(X)
      return A@self.a
  
  def score(self, X, y):
      '''
        RMSE
      '''
      y_pred = self.predict(X=X)
      return np.linalg.norm(y-y_pred, ord=2)/np.sqrt(X.shape[0])

