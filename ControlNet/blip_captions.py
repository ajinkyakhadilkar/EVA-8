import os
import cv2
import json
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_blip_caption(image_path, blip_model, blip_processor):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = blip_processor(image, return_tensors="pt").to("cuda")

    out = blip_model.generate(**inputs)
    decoded_caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return decoded_caption

def generate_prompt_json(source_folder, blip_model, prompt_file):
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    prompt_data = []

    # Iterate over the images in the source folder
    for i, filename in enumerate(os.listdir(source_folder)):
        print(str(i)+'/10000')
        # Check if it's an image file
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Compute Canny edges and get the destination path
            source_path = os.path.join(source_folder, filename)

            # Generate BLIP caption
            caption = generate_blip_caption(source_path, blip_model, blip_processor)

            # Create the prompt entry
            prompt_entry = {
                'source': filename,
                'target': filename,
                'prompt': caption
            }

            # Append to the prompt data list
            prompt_data.append(prompt_entry)

    # Write the prompt data to the JSON file
    with open(prompt_file, 'w') as f:
        json.dump(prompt_data, f, indent=4)

# Example usage
source_folder = 'target'
prompt_file = 'prompt.json'

# Load the BLIP model
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

generate_prompt_json(source_folder, blip_model, prompt_file)
