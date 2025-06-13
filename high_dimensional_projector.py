import numpy as np


class HighDimensionalProjector:

    def __init__(self):
        self.feature_weights = {
            'color': 0.3,
            'texture': 0.3,
            'edge': 0.2,
            'spatial': 0.2
        }

    def project_features(self, feature_dict):
        
        normalized_features = {}
        for key, features in feature_dict.items():
            if key != 'combined':
                
                norm = np.linalg.norm(features)
                normalized_features[key] = features / (norm + 1e-8)

        
        projection = np.concatenate([
            normalized_features['color'] * self.feature_weights['color'],
            normalized_features['texture'] * self.feature_weights['texture'],
            normalized_features['edge'] * self.feature_weights['edge'],
            normalized_features['spatial'] * self.feature_weights['spatial']
        ])

        return projection, normalized_features