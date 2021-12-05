import numpy as np
from utils import fgsm

def linear_solve(A, b):
    AAT = A@(A.T)
    # w = np.linalg.solve(AAT, b)
    w = np.linalg.lstsq(AAT, b)[0]
    x = (A.T)@w
    return x


class IterativeSinModel:
    '''
    H(n) :  h(x) = \sum_j [b_j*sin(a_0+a_1*x+a_2*x^2+...)]
    '''

    def __init__(self, n, MAX_DEGREE=3):
        self.D = MAX_DEGREE+1
        self.n = n # No. of Features
        self.F = np.random.rand(self.D, self.n) * 10 # Feature Tranformation matrix D X n
        self.a = None
    
    def generate_features(self, X):
        X = X.flatten()
        arrays = [X**i for i in range(self.D)] # includes constant
        A = np.stack(arrays, axis = 1)        # N_data X D
        A = A @ self.F                        # N_data X n
        A = np.sin(A)
        return A

    def feed(self, X):
        '''
        Feed Data to Store its Full Representation A
        '''
        self.A = self.generate_features(X)

    def feed_test(self, X):
        self.A_test = self.generate_features(X)

    def fit(self, X, Y, n_fit):
        '''
        X: (d,1)
        Y: (d,1)
        n_fit: No. of features to fit
        '''
        # self.a = linear_solve(A, Y)
        self.a = np.linalg.lstsq(self.A[:,:n_fit], Y)[0]       # N_data x n TODO: Faster
        return self

    def predict(self, n_fit, A):
        return (A[:,:n_fit])@self.a
    
    def score_test(self, y, n_fit):
        '''
            RMSE
        '''
        y_pred = self.predict(n_fit, self.A_test)
        return np.linalg.norm(y-y_pred, ord=2)/np.sqrt(y.shape[0])

    def score_train(self, y, n_fit):
        '''
            RMSE
        '''
        y_pred = self.predict(n_fit, self.A)
        return np.linalg.norm(y-y_pred, ord=2)/np.sqrt(y.shape[0])
    
    def score_adversarial(self, y, n_fit, epsilon):
        '''
            RMSE
        '''
        dt = lambda X : fgsm.loss_derivative_mse(X,y,self.a.reshape(-1,1))
        adversarial_data = fgsm.fgsm_adversarial_data(self.A_test[:,:n_fit], dt, epsilon)
        y_pred = self.predict(n_fit, adversarial_data)
        return np.linalg.norm(y-y_pred, ord=2)/np.sqrt(y.shape[0])

    
    
