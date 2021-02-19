import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    def __optimalPi(self, y):
        pi = [] 
        for j in range(3):
            pi.append(len([1 for i in y if i == j]))
        pi = [p / len(y) for p in pi]
        self.pi = pi 
        return 

    def __mean(self, X, y):
        self.__optimalPi(y)
        mu = [] 
        y = np.reshape(y, (1, X.shape[0]))
        for j in range(3):
            n = len([1 for i in y[0] if i == j])
            total = [0, 0]
            for i in range(len(X)):
                if y[0][i] == j:
                    total[0] += X[i][0] 
                    total[1] += X[i][1]
            mu.append(np.array(total) * (1 / n)) 
        return np.array(mu)

    def __covarianceMatrix(self, X, y):
        cov = []
        mu = self.__mean(X,y)
        X = np.array(X)
        for j in range(3):
            v = [0, 0]
            for i in range(len(X)):
                v[0] = (X[i][0] - mu[j][0]) * np.transpose(X[i][0] - mu[j][0])
                v[1] = (X[i][1] - mu[j][1]) * np.transpose(X[i][1] - mu[j][1])
            cov.append(np.diag(v))
        return np.array(cov) * (1 / len(X)) 

    # TODO: Implement this method!
    def fit(self, X, y):
        self.mu = self.__mean(X,y)
        self.cov = self.__covarianceMatrix(X,y)
        print("cov " + str(self.cov))
        print("mu " + str(self.mu))

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for i in range(len(X_pred)):
            stars = []
            for j in range(3):  
                stars.append(self.pi[j] * mvn.pdf(x=X_pred[i],mean=self.mu[j],cov=self.cov[j]))
            preds.append(stars.index(max(stars)))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        pass
