# Avataar_Assignment

## Assignment: Placing an Object’s Image in a Text-Conditioned Scene

### Author: Ananya

---

## Objective

This project aims to place an object image into a realistic, text-conditioned scene using generative AI techniques. Pre-trained models like Stable Diffusion are used for background generation, and YOLO is used for object and surface detection. The product is seamlessly integrated into the generated background.

---

## Key Goals:

- **Text-Conditioned Background**: The background aligns with the user’s text prompt (e.g., "A modern kitchen").
- **Unaltered Product**: The input product is resized without distortion.
- **Natural Placement**: The product is placed on detected surfaces (e.g., a table) without overlapping pre-existing objects.

---

## Setup and Installation

Install the required dependencies:

```bash
pip install transformers diffusers rembg moviepy matplotlib opencv-python
```

---

# Download the run.py file and use Google Colab to run it by setting runtime to T4 GPU.

Upload the file path.
Enter the text prompt to view the results.

---
For detailed step-by-step working - view the Avataar_H1.ipynb notebook.
---
## Process Overview:

### Upload a Product Image: 
The script will prompt you to upload an image.

### Remove Background:
 The background will be removed using the rembg library.

### Identify the Product:
 The product will be identified using a Vision Transformer model.

### Generate Background:
 A background will be generated based on your text input.

### Surface Detection Using Edge Detection and Contour Analysis:
 Detect natural surfaces for the product.

### Use YOLO for understanding the generated background: 
Ensure the product is placed intelligently without overlapping existing objects.

### Place the Product: 
The product will be placed intelligently in the generated background using surface and object detection.

### Display the Final Result: 
The output will be displayed as an image.

## Approach

### Background Removal: 
We use rembg to remove the product’s background.

### Product Identification:
 A pre-trained Vision Transformer (vit-base-patch16-224) is used for classifying the product.

### Background Generation: 
The Stable Diffusion pipeline generates a background scene based on a text prompt.

### Surface and Object Detection:
 Edge detection with OpenCV and YOLOv5 is used to detect surfaces and objects, ensuring the product is placed naturally in the scene.

### Intelligent Placement: 
The product is resized and placed on a logical surface, avoiding overlap with objects in the background.

---

## How to Improve

### Enhanced Surface Detection:
 Implement more precise surface detection models to improve product placement.

### Dynamic Scene Generation:
 Expand the project to support dynamic, user-interactive scenes.

---

# Sample Images

![image](https://github.com/user-attachments/assets/088ccde1-2dd0-4313-bd71-ced05d9e1469)
![image](https://github.com/user-attachments/assets/1298ce21-e597-416d-8365-7c67e8c20222)
![image](https://github.com/user-attachments/assets/62d9dcd6-71e1-49df-b1cd-2a90819d25ed)
![image](https://github.com/user-attachments/assets/b5877052-f9b4-427a-ba2c-258258bd1e5a)

