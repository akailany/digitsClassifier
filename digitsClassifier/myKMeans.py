# Atiya Kailany
# CS 679 - Machine Learning
# April 25, 2021

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclid_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class myKMeans():

    def __init__(self, K=5, iteration=10):
        # Constructor
        # setting local environment variables. self = this (in java)
        self.K = K
        self.iteration = iteration

        # intialize empty clusters based on passed K variable
        self.clusters = [[] for _ in range(self.K)]
        
        # intialize empty centers variable
        self.centers = []

    def predict(self, data):
        self.data = data
        self.features, self.labels = data.shape

        # initialize
        features_indices = np.random.choice(self.features, self.K, replace=False)
        self.centers = [self.data[index] for index in features_indices]

        # Iterate 10 times or based on user input
        for _ in range(self.iteration):

            # Assign labels to closest centers
            self.clusters = self.calculate_clusters(self.centers)

            # Calculate new centers
            prev_centers = self.centers
            self.centers = self.calculate_centers(self.clusters)

            # check convergance
            if self.check_convergance(prev_centers, self.centers):
                break


        # Classify
        return self.classify_cluster_labels(self.clusters)


    def classify_cluster_labels(self, clusters):
        # each class will be classified based on the cluster it was put in
        labels = np.empty(self.features)

        for cluster_index, cluster in enumerate(clusters):
            for label_index in cluster:
                labels[label_index] = cluster_index
        return labels

    def calculate_clusters(self, centers):
        # classify into a cluster based on the closest center
        clusters = [[] for _ in range(self.K)]
        for index, label in enumerate(self.data):
            center_index = self.closest_centers(label, centers)
            clusters[center_index].append(index)
        return clusters

    def closest_centers(self, label, centers):
        # euclidean distance label to center
        distances = [euclid_dist(label, data_point) for data_point in centers]
        closest_index = np.argmin(distances)
        return closest_index

    def calculate_centers(self, clusters):
        # average cluster and assign the average as the center
        centers = np.zeros((self.K, self.labels))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0)
            centers[cluster_index] = cluster_mean
        return centers

    def check_convergance(self, prev_centers, centers):
        # basically check distance between old and new centers
        distances = [euclid_dist(prev_centers[i], centers[i]) for i in range(self.K)]
        return sum(distances) == 0

    def distance(self, point, center):
        square_sums = 0.0
        for point_i, center_i in zip(point, center):
            square_sums += (point_i - center_i) ** 2
        return np.sqrt(square_sums)

    def sumSquaredError(self, data, k):
        error = 0
        for i in range(k):
            cluster = self.clusters[i]
            center = self.centers[i]
            for data_point_index in cluster:
                datapoint = data[data_point_index]
                error += self.distance(datapoint, center) ** 2
        return error/100000



    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.data[index].T
            ax.scatter(*point)

        for point in self.centers:
            ax.scatter(*point, marker="x", s=800, color='black', linewidth=3)

        plt.show()

