import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from image_feature_extractor import ImageFeatureExtractor
from high_dimensional_projector import HighDimensionalProjector
from similarity_calculator import SimilarityCalculator


class ImageMatcher:

    def __init__(self, database_path):
        self.database_path = database_path
        self.extractor = ImageFeatureExtractor()
        self.projector = HighDimensionalProjector()
        self.calculator = SimilarityCalculator()
        self.database_features = {}
        self.image_paths = []

    def load_database(self, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp']):
        print("Loading database images...")

        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.database_path, ext)))
            self.image_paths.extend(glob.glob(os.path.join(self.database_path, ext.upper())))

        print(f"Found {len(self.image_paths)} images")

        for i, img_path in enumerate(self.image_paths):
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    
                    image = cv2.resize(image, (256, 256))

                    
                    features = self.extractor.extract_all_features(image)
                    projection, normalized_features = self.projector.project_features(features)

                    self.database_features[img_path] = {
                        'projection': projection,
                        'features': normalized_features,
                        'raw_features': features
                    }

                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(self.image_paths)} images")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        print(f"Successfully loaded {len(self.database_features)} images")

    def find_similar_images(self, query_image_path, top_k=5):
        
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise ValueError(f"Cannot load image: {query_image_path}")

        query_image = cv2.resize(query_image, (256, 256))
        query_features = self.extractor.extract_all_features(query_image)
        query_projection, query_normalized = self.projector.project_features(query_features)

        
        similarities = []
        detailed_similarities = []

        for db_path, db_data in self.database_features.items():
            
            main_sim = self.calculator.cosine_similarity(query_projection, db_data['projection'])

            
            weighted_sim, feature_sims = self.calculator.weighted_similarity(
                query_normalized, db_data['features']
            )

            similarities.append(main_sim)
            detailed_similarities.append({
                'path': db_path,
                'similarity': main_sim,
                'weighted_similarity': weighted_sim,
                'feature_breakdown': feature_sims
            })

        
        probabilities = self.calculator.calculate_probabilities(similarities)

        
        for i, detail in enumerate(detailed_similarities):
            detail['probability'] = probabilities[i]

        
        detailed_similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return detailed_similarities[:top_k], query_features

    def display_results(self, results, query_path, query_features):
        print("=" * 80)
        print("IMAGE SIMILARITY ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Query Image: {os.path.basename(query_path)}")
        print(f"Database Size: {len(self.database_features)} images")
        print(f"Feature Vector Dimensions: {len(results[0]['feature_breakdown']) * 4}")
        print()

        
        print("TOP MATCHES:")
        print("-" * 50)

        for i, result in enumerate(results, 1):
            print(f"{i}. {os.path.basename(result['path'])}")
            print(f"   Overall Similarity: {result['similarity']:.4f}")
            print(f"   Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
            print(f"   Feature Breakdown:")
            for feature_type, sim in result['feature_breakdown'].items():
                print(f"     - {feature_type.capitalize()}: {sim:.4f}")
            print()

        
        best_match = results[0]
        print("WHY THIS IMAGE WAS CHOSEN (Mathematical Analysis):")
        print("-" * 60)
        print(f"Best Match: {os.path.basename(best_match['path'])}")
        print(f"Cosine Similarity Score: {best_match['similarity']:.6f}")
        print(f"This means the angle between feature vectors is: {np.arccos(best_match['similarity']) * 180/np.pi:.2f}°")
        print()
        print("Feature Analysis:")
        for feature_type, sim in best_match['feature_breakdown'].items():
            contribution = sim * self.projector.feature_weights[feature_type]
            print(f"- {feature_type.capitalize()} similarity: {sim:.4f} (weighted contribution: {contribution:.4f})")
        print()
        print("Probability Calculation:")
        print(f"- Raw similarity: {best_match['similarity']:.4f}")
        print(f"- After softmax normalization: {best_match['probability']:.4f}")
        print(f"- Confidence level: {best_match['probability']*100:.2f}%")

        
        self.plot_results(results[:3], query_path)

    def plot_results(self, results, query_path):
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(20, 5))

        
        query_img = cv2.imread(query_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query Image\n{os.path.basename(query_path)}", fontsize=10)
        axes[0].axis('off')

        
        for i, result in enumerate(results, 1):
            img = cv2.imread(result['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(f"Match {i}\n{os.path.basename(result['path'])}\n"
                              f"Similarity: {result['similarity']:.3f}\n"
                              f"Probability: {result['probability']*100:.1f}%", fontsize=9)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()