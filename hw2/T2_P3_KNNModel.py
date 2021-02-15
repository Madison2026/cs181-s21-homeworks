import numpy as np
# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = [0 for v in range(len(X_pred))]
        for i in range(len(X_pred)):
            distances = {} 
            for j in range(len(self.X)):
                d = ((X_pred[i][0] - self.X[j][0]) / 3) ** 2 + (X_pred[i][1] - self.X[j][1]) ** 2
                distances[j] = d
            sorted_indices = dict(sorted(distances.items(), key=lambda x: x[1]))
            stars = [0, 0, 0]
            for m in range(self.K):
                star_name = self.y[list(sorted_indices.keys())[m]]
                stars[star_name] += 1
            preds[i] = stars.index(max(stars))
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y