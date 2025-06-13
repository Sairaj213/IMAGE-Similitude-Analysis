import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_matcher import ImageMatcher

st.sidebar.title("Image Similarity Matcher")
st.sidebar.markdown(
    """
    **Steps to follow:**
    1. Upload your database images.
    2. Upload your query image.
    3. Choose how many top matches you want.
    4. Click "Run Analysis".
    """
)

db_files = st.sidebar.file_uploader(
    "Upload Database Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
)
query_file = st.sidebar.file_uploader(
    "Upload Query Image", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=False
)
top_k = st.sidebar.number_input("Top K Matches", min_value=1, max_value=20, value=5, step=1)
run_button = st.sidebar.button("Run Analysis")

if run_button:
    if not db_files:
        st.error("Please upload at least one image to build the database.")
    elif not query_file:
        st.error("Please upload a query image.")
    else:
        st.info("Processing images, please wait...")

        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in db_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            query_temp_path = os.path.join(temp_dir, query_file.name)
            with open(query_temp_path, "wb") as f:
                f.write(query_file.getbuffer())

            matcher = ImageMatcher(temp_dir)
            matcher.load_database()

            if len(matcher.database_features) == 0:
                st.error("No valid images were found in the provided database!")
            else:
                results, query_features = matcher.find_similar_images(query_temp_path, top_k)

                st.subheader("Analysis Results")
                st.write(f"**Query Image:** {query_file.name}")
                st.write(f"**Database Size:** {len(matcher.database_features)} images")

                for idx, res in enumerate(results, start=1):
                    st.markdown(f"### Match {idx}: {os.path.basename(res['path'])}")
                    st.write(f"**Overall Similarity:** {res['similarity']:.4f}")
                    st.write(f"**Probability:** {res['probability']*100:.2f}%")
                    st.write("**Feature Breakdown:**")
                    for feature_type, sim in res["feature_breakdown"].items():
                        st.write(f"- {feature_type.capitalize()}: {sim:.4f}")

                
                num_plot = min(3, len(results))
                fig, axes = plt.subplots(1, num_plot + 1, figsize=(15, 5))

                q_img = cv2.imread(query_temp_path)
                if q_img is not None:
                    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
                    axes[0].imshow(q_img)
                    axes[0].axis("off")
                    axes[0].set_title("Query Image")
                else:
                    axes[0].text(0.5, 0.5, "Query image not found.", ha="center", va="center")
                    axes[0].axis("off")

                for i in range(num_plot):
                    db_img_path = results[i]['path']
                    db_img = cv2.imread(db_img_path)
                    if db_img is not None:
                        db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                        axes[i + 1].imshow(db_img)
                        axes[i + 1].axis("off")
                        axes[i + 1].set_title(
                            f"Match {i+1}\nSim: {results[i]['similarity']:.3f}\nProb: {results[i]['probability']*100:.1f}%"
                        )
                    else:
                        axes[i + 1].text(0.5, 0.5, "Image not found", ha="center", va="center")
                        axes[i + 1].axis("off")

                st.pyplot(fig)
                st.success("Analysis completed!")
