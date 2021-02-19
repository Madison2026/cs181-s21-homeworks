import numpy as np
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.soft = []
        self.grads = []

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
        runs = 200000
        self.X = X
        self.y = y
        self.W = np.random.rand(3, X.shape[1])
        self.grads = []
        for i in range(runs):
            self.W = self.W + (self.eta * self.__calculateError(X, y))
            self.grads.append(self.W)

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        self.X_pred = X_pred
        self.soft = []
        preds = []
        for i in range(len(X_pred)):
            z = self.W @ X_pred[i] 
            denom = np.sum(np.exp(z))
            softmax = list(np.exp(z) / denom) 
            self.soft.append(max(softmax))
            preds.append(softmax.index(max(softmax)))
        return np.array(preds)
  
    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        pass
        """
        params = [.05, .01, .001]
        for l in params:
            for e in params:
                self.eta = e
                self.lam = l
                self.fit(self.X, self.y)
                losses = []
                for g in self.grads:
                    self.W = g
                    self.predict(self.X_pred)               
                    loss = [-1 * np.log(s + .001) for s in self.soft]
                    losses.append(sum(loss))
                fig, ax = plt.subplots() 
                ax.set_xlabel('Number of Iterations')  
                ax.set_ylabel('Negative Log Liklihood') 
                ax.set_title("Negative Log Liklihood Loss, Lamda=" + str(self.lam) + ",eta=" + str(self.eta))  
                ax.plot(range(200000), losses)  
                plt.savefig(output_file + "Lamda=" + str(self.lam) + ",eta=" + str(self.eta)+ '.png')
                if show_charts:
                    plt.show()
        """ 
        
