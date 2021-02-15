import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.W = np.array([])

    def __calculateError(self, x, y):
        gradient = np.zeros((x.shape[1], 1))
        for y_i, y_hat, x_i in zip(y, self.predict(x), x):
            error = np.array([((y_i - y_hat) * x_i)])
            gradient = gradient + error.T   
        gradient = gradient / len(x) 
        #return gradient
        self.W = gradient
        return gradient + self.lam * np.linalg.norm(gradient)[0] ** 2 + self.lam * np.linalg.norm(gradient)[0] 

    # TODO: Implement this method!
    def fit(self, X, y):
        runs = 20000
        self.W = (self.eta * self.__calculateError(X, y))
        for i in range(runs):
            self.W = self.W + (self.eta * self.__calculateError(X, y))

        print("fit W" + str(self.W))

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        softmax = []
        for i in range(len(X_pred)):
            print ("W predict" + str(self.W))
            z = self.W @ X_pred[i]
            denom = 0
            for j in range(len(z)): 
                denom += np.exp(z[j])
            softmax.append(z / denom) 
        preds = [softmax[i].index(max(softmax[i])) for i in softmax] 
        """
        preds = []
        for x in X_pred:
            z = np.cos(x ** 2).sum()
            preds.append(1 + np.sign(z) * (np.abs(z) > 0.3))
         """
        return np.array(preds)
       
    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        pass
