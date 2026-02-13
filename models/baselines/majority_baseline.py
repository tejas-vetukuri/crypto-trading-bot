# models/baselines/majority_baseline.py
import numpy as np
from collections import Counter


class MajorityBaseline:
    def fit(self, X, y):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.majority_class] * len(X))
