from PIL import Image
import os

# Set input and output paths
input_folder = 'F:/2024 project group/24ACK006 - seed detection and fertilizer detection/100% project/new'
output_folder = 'F:/2024 project group/24ACK006 - seed detection and fertilizer detection/100% project/new1'

# Set the desired image size
img_width, img_height = 64, 64

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop over each file in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Open, resize, and save the image
    try:
        with Image.open(file_path) as img:
            resized_img = img.resize((img_width, img_height))
            rgb_img = resized_img.convert("RGB")
            rgb_img.save(os.path.join(output_folder, filename), "PNG")
            print(f"Processed and saved: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
