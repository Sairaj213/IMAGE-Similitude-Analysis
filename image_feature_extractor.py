import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycoprops, graycomatrix
from skimage.filters import gabor
import warnings

warnings.filterwarnings('ignore')


class ImageFeatureExtractor:

    def __init__(self):
        self.scaler = StandardScaler()

    def extract_color_histogram(self, image, bins=64):
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        
        hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256])
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])

        
        features = np.concatenate([
            hist_b.flatten() / np.sum(hist_b),
            hist_g.flatten() / np.sum(hist_g),
            hist_r.flatten() / np.sum(hist_r),
            hist_h.flatten() / np.sum(hist_h),
            hist_s.flatten() / np.sum(hist_s)
        ])

        return features

    def extract_texture_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / np.sum(lbp_hist)

        
        gabor_features = []
        for theta in [0, 45, 90, 135]:
            real, _ = gabor(gray, frequency=0.1, theta=np.deg2rad(theta))
            gabor_features.extend([np.mean(real), np.std(real)])

        
        glcm = graycomatrix(gray.astype(np.uint8), [1], [0, 45, 90, 135],
                             levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()

        features = np.concatenate([lbp_hist, gabor_features, contrast, homogeneity])
        return features

    def extract_edge_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        
        canny = cv2.Canny(gray, 50, 150)

        
        features = [
            np.mean(sobel_magnitude),
            np.std(sobel_magnitude),
            np.sum(canny > 0) / (canny.shape[0] * canny.shape[1]),  
            np.mean(sobel_x[sobel_x > 0]) if np.any(sobel_x > 0) else 0,  
            np.mean(sobel_y[sobel_y > 0]) if np.any(sobel_y > 0) else 0,  
        ]

        return np.array(features)

    def extract_spatial_moments(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        moments = cv2.moments(gray)

        
        hu_moments = cv2.HuMoments(moments).flatten()

        
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = 0, 0

        
        features = np.concatenate([
            hu_moments,
            [cx, cy, np.mean(gray), np.std(gray), np.var(gray)]
        ])

        return features

    def extract_all_features(self, image):
        color_features = self.extract_color_histogram(image)
        texture_features = self.extract_texture_features(image)
        edge_features = self.extract_edge_features(image)
        spatial_features = self.extract_spatial_moments(image)

        
        all_features = np.concatenate([
            color_features,
            texture_features,
            edge_features,
            spatial_features
        ])

        return {
            'combined': all_features,
            'color': color_features,
            'texture': texture_features,
            'edge': edge_features,
            'spatial': spatial_features
        }