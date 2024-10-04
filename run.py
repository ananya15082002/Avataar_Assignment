import torch
from PIL import Image
from rembg import remove
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np
from IPython.display import display, Image as IPImage

# Function to remove background from the product image
def remove_background(input_image_path, output_image_path):
    with open(input_image_path, "rb") as input_file:
        input_image = input_file.read()
    output_image = remove(input_image)
    
    # Save the image as PNG to preserve transparency
    with open(output_image_path, "wb") as output_file:
        output_file.write(output_image)

    # Ensure the output image has an alpha channel (transparency)
    product = Image.open(output_image_path)
    if product.mode != 'RGBA':
        product = product.convert('RGBA')
    product.save(output_image_path)
    return product

# Function to identify the product using image classification
def identify_product(input_image_path):
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    result = classifier(input_image_path)
    product_label = result[0]['label']
    print(f"Identified product: {product_label}")
    return product_label

# Function to generate background based on the user’s text prompt
def generate_background(text_prompt, output_background_path):
    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    generated_image = model(prompt=text_prompt).images[0]
    generated_image.save(output_background_path)
    return generated_image

# Function to detect surfaces in the generated background
def detect_surfaces(background_path):
    background_image = cv2.imread(background_path)
    gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(background_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("/content/debug_contours.png", background_image)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

# YOLOv5 Object Detection
def detect_objects_with_yolov5(background_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    results = model(background_path)
    objects = results.pandas().xyxy[0]
    return objects

# Function to place the product intelligently in the generated background
def place_product_intelligently(background_path, product_path, output_composite_path):
    background = Image.open(background_path).convert("RGB")
    product = Image.open(product_path).convert("RGBA")
    
    surface_box = detect_surfaces(background_path)
    detected_objects = detect_objects_with_yolov5(background_path)

    product_position = (int(background.width * 0.5), int(background.height * 0.7))
    
    if surface_box:
        x, y, w, h = surface_box
        product_width = int(w * 0.9)
        aspect_ratio = product.width / product.height
        product_height = int(product_width / aspect_ratio)
        product = product.resize((product_width, product_height), Image.LANCZOS)
        
        for _, obj in detected_objects.iterrows():
            obj_xmin, obj_ymin, obj_xmax, obj_ymax = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            if not (x < obj_xmin < x + w and y < obj_ymin < y + h):
                product_position = (x + 10, y + 10)
                break
        else:
            product_position = (int(background.width * 0.5), int(background.height * 0.7))

    background.paste(product, product_position, mask=product.split()[3])
    background.save(output_composite_path)
    return background

# Main function to perform all steps
def process_image_and_generate_result():
    while True:
        # Step 1: Get product image input or exit command
        product_image_path = input("Enter the product image file path (or type 'exit' to stop): ")
        if product_image_path.lower() == 'exit':
            print("Exiting program.")
            break

        # Step 2: Remove background from the product image
        product_image = remove_background(product_image_path, "/content/extracted_product.png")

        # Step 3: Identify the product
        product_name = identify_product("/content/extracted_product.png")
        
        # Step 4: Get user input for text prompt for background generation
        text_prompt = input("Enter the text prompt for the background (or type 'exit' to stop): ")
        if text_prompt.lower() == 'exit':
            print("Exiting program.")
            break
        
        # Step 5: Generate background
        background_image = generate_background(text_prompt, "/content/generated_background.png")

        # Step 6: Intelligently place the product in the generated background
        final_composite_image = place_product_intelligently("/content/generated_background.png", "/content/extracted_product.png", "/content/final_composite_intelligent.png")
        
        # Step 7: Display the final composite image
        display(IPImage("/content/final_composite_intelligent.png"))

# Call the main function
process_image_and_generate_result()
