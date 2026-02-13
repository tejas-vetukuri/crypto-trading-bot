# models/baselines/random_baseline.py
import numpy as np


class RandomBaseline:
    def fit(self, X, y):
        self.classes = np.unique(y)

    def predict(self, X):
        return np.random.choice(self.classes, size=len(X))
