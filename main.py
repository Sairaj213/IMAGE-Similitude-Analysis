import os
from image_matcher import ImageMatcher


def run_image_similarity_matcher():
    print("🔍 IMAGE SIMILARITY MATCHER - STATIC MATHEMATICAL APPROACH")
    print("=" * 70)

    database_path = input("Enter path to image database folder: ").strip()
    if not os.path.exists(database_path):
        print("❌ Database path does not exist!")
        return
    if not os.path.isdir(database_path):
        print("❌ Database path must be a directory!")
        return

    query_image_path = input("Enter path to query image file: ").strip()  
    if not os.path.exists(query_image_path):
        print("❌ Query image path does not exist!")
        return
    
    if not os.path.isfile(query_image_path):
        print("❌ Query path must be a file!")
        return

    top_k = input("Number of similar images to find (default 5): ").strip()
    top_k = int(top_k) if top_k.isdigit() else 5

    print("\n🚀 Starting analysis...")

    
    matcher = ImageMatcher(database_path)

    
    matcher.load_database()

    if len(matcher.database_features) == 0:
        print("❌ No images found in database!")
        return

    print(f"\n🔎 Analyzing query image: {os.path.basename(query_image_path)}")
    results, query_features = matcher.find_similar_images(query_image_path, top_k)

    
    matcher.display_results(results, query_image_path, query_features)

    print("\n✅ Analysis completed!")

    
    another = input("\nAnalyze another image? (y/n): ").strip().lower()
    if another == 'y':
        query_image_path = input("Enter path to new query image file: ").strip()  
        if os.path.exists(query_image_path) and os.path.isfile(query_image_path):  
            results, query_features = matcher.find_similar_images(query_image_path, top_k)
            matcher.display_results(results, query_image_path, query_features)
        else:
            print("❌ Invalid query image path!")

if __name__ == "__main__":
    run_image_similarity_matcher()