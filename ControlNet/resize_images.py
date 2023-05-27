import os
import cv2

# Set the path to the folder containing the images
folder_path = "destination"

# Create a list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Iterate over each image file
for image_file in image_files:
    # Read the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    # Resize the image to 512x512
    resized_image = cv2.resize(image, (512, 512))
    
    # Save the resized image
    cv2.imwrite(image_path, resized_image)

    print(f"Resized image: {image_file}")

print("Image resizing completed.")
