# 📷 Image Similarity Check - Static Mathematical Approach
<br> 
A project pipeline that performs image comparisons using mathematical features rather than machine learning, it extracts the features such as color histograms, edge information, spatial moments, these features into a weighted high-dimensional space and computes similarity scores using measures like cosine similarity to determine how similar images are.
<br>

## Project Definition
<br>

Instead of relying on deep learning models, this system uses classical computer vision and statistical methods to extract and process image features. The project includes:
- Feature extraction using methods like Local Binary Patterns (LBP), Gabor filters, and Gray Level Co-occurrence Matrix (GLCM).
- Projection of features into a weighted high-dimensional space.
- Similarity assessment using cosine similarity and a weighted similarity approach.
- A detailed breakdown and visual explanation of the matching process.
<br>

## Project Structure
<br>

```markdown-tree
Image_Similitude_Analysis/
├── image_feature_extractor.py    # Extracts various features (color, texture, edge, spatial) from images.
├── high_dimensional_projector.py # Projects features into a weighted high-dimensional space.
├── similarity_calculator.py      # Contains similarity functions (cosine and Euclidean) and probability conversion.
├── image_matcher.py              # Integrates the above modules: loads the image database, processes query images, and outputs similarity results.
├── main.py                       # Command-line interface to run the image similarity matcher interactively. 
├── streamlit_app.py              # Streamlit interface
└── README.md                     # Project documentation
```
<br>


## Function Definitions
<br>

### 1. `image_feature_extractor.py`
- **ImageFeatureExtractor**
  - `extract_color_histogram(image, bins=64)`: Computes color histograms in BGR, HSV (H and S channels) color spaces.
  - `extract_texture_features(image)`: Uses Local Binary Pattern, Gabor filters, and GLCM to extract texture details.
  - `extract_edge_features(image)`: Extracts edge characteristics using Sobel and Canny operators.
  - `extract_spatial_moments(image)`: Calculates spatial moments including Hu moments, centroids, and statistical metrics.
  - `extract_all_features(image)`: Aggregates all the above features into a consolidated feature dictionary.

### 2. `high_dimensional_projector.py`
- **HighDimensionalProjector**
  - `project_features(feature_dict)`: Normalizes individual feature sets (using L2 normalization) and projects them into a combined, weighted high-dimensional vector.

### 3. `similarity_calculator.py`
- **SimilarityCalculator**
  - `cosine_similarity(vec1, vec2)`: Computes cosine similarity between two feature vectors.
  - `euclidean_similarity(vec1, vec2)`: Computes a normalized Euclidean similarity.
  - `weighted_similarity(features1, features2, weights=None)`: Calculates a weighted average similarity across different feature types.
  - `calculate_probabilities(similarities)`: Converts a set of similarity scores into a probability distribution using a softmax function.

### 4. `image_matcher.py`
- **ImageMatcher**
  - `load_database(extensions)`: Scans a given folder for images, processes them to extract features, and stores the features.
  - `find_similar_images(query_image_path, top_k=5)`: Processes a query image, computes similarities with the database, and retrieves the top matching images.
  - `display_results(results, query_path, query_features)`: Outputs the details of the image comparisons (similarity scores, probability, and feature breakdown) and shows a Matplotlib-based plot.
  - `plot_results(results, query_path)`: Visualizes the query image alongside its top matches in a simple grid layout.

### 5. `main.py`
- Contains the `run_image_similarity_matcher()` function:
  - Provides a command-line interface for users.
  - Asks for the image database path, query image path, and number of similar images to retrieve.
  - Executes the matching process and displays results interactively in the terminal/command prompt.

### 6. `streamlit_app.py`
- Provides UI using Streamlit:
  - **Sidebar:** Contains clear instructions, drag-and-drop upload widgets for the image database and a separate uploader for the query image, and settings (like the number of matches to display).
  - **Main Area:** Displays detailed analysis including similarity scores, a breakdown per feature type, and visual comparison plots.
<br>    

## Project Flow
<br>

1. **Database Setup:**  
   - The user provides a folder containing image files (or uploads images via the Streamlit UI).
   - The system resizes images and extracts various mathematical features from each image.

2. **Query Image Processing:**  
   - A query image is provided by the user (via command-line input or via the Streamlit uploader).
   - The query image goes through the same preprocessing and feature extraction pipeline.

3. **Feature Projection & Similarity Calculation:**
   - Features are normalized and each feature type is weighted appropriately to form a high-dimensional projection.
   - The similarity between the query image and each image in the database is computed using cosine similarity.
   - A softmax function converts these similarity scores into probabilities, allowing for a probabilistic interpretation of “match quality.”

4. **Output and Analysis:**  
   - The system outputs a detailed analysis, including overall similarity scores, a probability distribution, and a breakdown of similarities by feature type.
   - Visual comparisons of the query image and top matching images are provided (using Matplotlib in CLI mode and rendered within the Streamlit UI).
<br>     

## User Instructions
<br>

### Command-line Mode

1. **Preparation:**
   - Clone the repository.
   - Ensure you have Python 3 installed.
     
2. **Execution:**
   - Prepare a folder containing your database images (supported formats: JPG, JPEG, PNG, BMP).
   - Run the main script:
     ```bash
     python main.py
     ```
   - Follow the instructions to input the paths for the image database and query image and select the number of similar images.
   - View the detailed output in the terminal along with plots.
<br> 

### Streamlit UI Mode
<br>

1. **Preparation:**
   - Clone the repository.
     
2. **Execution:**
   - Run the Streamlit application:
     ```bash
     streamlit run streamlit_app.py
     ```
3. **Usage:**
   - In the sidebar, drag and drop your image database files.
   - Upload the query image.
   - Adjust settings such as the number of images to retrieve.
   - Click the **Run Analysis** button.
   - The main area will display the detailed analysis along with visual comparisons between the query and matching images.
