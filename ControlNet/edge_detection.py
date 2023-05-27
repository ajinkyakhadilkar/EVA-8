import os
import cv2

# Set the path to the source and destination folders
source_folder = "target"
destination_folder = "source"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get the list of image files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Iterate over each image file
for image_file in image_files:
    # Read the image
    image_path = os.path.join(source_folder, image_file)
    image = cv2.imread(image_path)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)  # Adjust the threshold values as needed
    
    # Save the edge image to the destination folder
    destination_path = os.path.join(destination_folder, image_file)
    cv2.imwrite(destination_path, edges)

    print(f"Processed image: {image_file}")

print("Edge detection completed.")