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

    def __calculateError(self, x, y):
        gradient = np.zeros((3, x.shape[1]))
        for j in range(3):
            count = 0
            g = np.zeros((1, x.shape[1]))
            for y_i, y_hat, x_i in zip(y, self.predict(x), x):
                if (y_i == j):
                    count += 1
                    error = np.array([((y_i - y_hat) * x_i)])
                    g = g + error.T 
            gradient[j] = (g / count)[:1]
        self.W = gradient + self.lam * np.linalg.norm(gradient) ** 2 
        return self.W

    # TODO: Implement this method!
    def fit(self, X, y):
        runs = 20000
        self.W = np.random.rand(3, X.shape[1])
        for i in range(runs):
            self.W = self.W + (self.eta * self.__calculateError(X, y))
        print("w: " + str(self.W))

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for i in range(len(X_pred)):
            z = self.W @ X_pred[i] 
            print("z " + str(z))
            denom = np.sum(np.exp(z))
            
            softmax = np.exp(z) / denom 
            softmax = list(softmax)
            preds.append(softmax.index(max(softmax)))
        return np.array(preds)
       
    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        pass
