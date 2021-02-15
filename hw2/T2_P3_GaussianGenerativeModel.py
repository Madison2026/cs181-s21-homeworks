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
            pi.append(len([1 for i in y[0] if i == j]))
        return pi

    def __mean(self, X, y):
        mu = [] 
        y = np.reshape(y, (1, X.shape[0]))
        s = sum(y @ X) 
        for j in range(3):
            n = len([1 for i in y[0] if i == j])
            mu.append(s * (1 / n)) 
        return mu

    def __covarianceMatrix(self, X, y):
        cov = []
        mu = self.__mean(X,y)
        for i in range(len(X)):
            total = 0 
            for j in range(3):
                total += (X[i] - mu[j]) @ (X[i] - mu[j]).T * y[i]
            cov.append(total)
            print("total " + str(total))
        return (1 / len(X)) * np.array(cov)
        #return X.T

    # TODO: Implement this method!
    def fit(self, X, y):
        print("mean" + str(self.__mean(X,y)))
        print("cov" + str(self.__covarianceMatrix(X,y)))
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.

        preds = []
        for x in X_pred:
            z = np.sin(x ** 2).sum()
            preds.append(1 + np.sign(z) * (np.abs(z) > 0.3))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        pass
