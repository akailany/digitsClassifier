# Atiya Kailany
# CS 679 - Machine Learning
# April 25, 2021

import numpy as np

class myPCA:

    def __init__(self, n_features):

        # Constructor
        self.n_features = n_features
        self.features = None
        self.mean = None

    def fit(self, data):

        # mean
        self.mean = np.mean(data, axis=0)
        data = data - self.mean

        cov = np.cov(data.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # sort in descending order. So most important feature is at the top
        index = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[index]

        self.features = eigenvectors[0:self.n_features]

    def transform(self, data):
        data = data - self.mean
        return -1 * np.dot(data, self.features.T)