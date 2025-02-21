import os
import random
import shutil
from tqdm import tqdm

# Define paths
base_folder = "evaluation-2/datasets/substitute-dataset"
output_folder = "evaluation-2/datasets/subsets"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get category folders
categories = [cat for cat in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, cat))]

# Collect all images with category labels
all_images_by_category = {}
for category in categories:
    category_path = os.path.join(base_folder, category)
    all_images_by_category[category] = os.listdir(category_path)
    random.shuffle(all_images_by_category[category])  # Shuffle for random selection

# Flatten the category-wise images into a single list of (category, image)
all_images = []
for category, images in all_images_by_category.items():
    all_images.extend([(category, image) for image in images])
random.shuffle(all_images)  # Shuffle for random selection across all categories

# Initialize variables
all_selected_images = set()  # Store all images selected across subsets
previous_subset_images = set()  # Store images from the previous subset

# Create 20 subsets
for i in tqdm(range(1, 21), desc="Creating subsets"):
    # Number of new images to add in this step
    images_to_add = 1000
    new_images = []  # Temporary list for this subset's new images
    
    # Randomly select 1,000 new unique images
    while len(new_images) < images_to_add and all_images:
        category, new_image = all_images.pop()
        if (category, new_image) not in all_selected_images:
            new_images.append((category, new_image))
            all_selected_images.add((category, new_image))
    
    # Combine previous subset images with the newly selected ones
    current_subset_images = previous_subset_images.union(new_images)

    # Create subset folder
    subset_folder = os.path.join(output_folder, f"subset_{i}")
    os.makedirs(subset_folder, exist_ok=True)
    
    # Copy images into the subset folder, maintaining structure
    for category, image in tqdm(current_subset_images, desc=f"Copying images for subset_{i}", leave=False):
        category_subset_folder = os.path.join(subset_folder, category)
        os.makedirs(category_subset_folder, exist_ok=True)
        src_path = os.path.join(base_folder, category, image)
        dst_path = os.path.join(category_subset_folder, image)
        shutil.copy(src_path, dst_path)
    
    # Update previous_subset_images for the next iteration
    previous_subset_images = current_subset_images

    tqdm.write(f"Created subset_{i} with {len(current_subset_images)} total images.")
