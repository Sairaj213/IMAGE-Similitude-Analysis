import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.special import softmax


class SimilarityCalculator:

    def __init__(self):
        pass

    def cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)

    def euclidean_similarity(self, vec1, vec2):
        distance = euclidean(vec1, vec2)
        return 1 / (1 + distance)

    def weighted_similarity(self, features1, features2, weights=None):
        if weights is None:
            weights = {'color': 0.3, 'texture': 0.3, 'edge': 0.2, 'spatial': 0.2}

        similarities = {}
        total_similarity = 0

        for feature_type in ['color', 'texture', 'edge', 'spatial']:
            if feature_type in features1 and feature_type in features2:
                sim = self.cosine_similarity(features1[feature_type], features2[feature_type])
                similarities[feature_type] = sim
                total_similarity += sim * weights[feature_type]

        return total_similarity, similarities

    def calculate_probabilities(self, similarities):
        similarities = np.array(similarities)
        probabilities = softmax(similarities * 10) 
        return probabilities