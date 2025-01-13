from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from collections import Counter
import openai
import os
import hashlib
import logging

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Set your OpenAI API key directly (NOT RECOMMENDED for production)
openai.api_key = "your-openai-api-key"

html_element_descriptions = {
    "button": "A clickable button, often used for submitting forms or triggering actions.",
    "input": "A text input field for the user to enter data, such as name or email.",
    "link": "A clickable link that navigates to another page.",
    "image": "An image displayed on the webpage.",
    "navbar": "A navigation bar containing links to important sections of the site.",
}

# Basic color mappings
COLOR_NAMES = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}

# Function to find the closest color name
def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None
    for name, rgb in COLOR_NAMES.items():
        distance = sum((requested_color[i] - rgb[i]) ** 2 for i in range(3))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

# Function to extract the dominant color
def get_dominant_color(image):
    image = image.convert("RGB").resize((150, 150))
    pixels = list(image.getdata())
    most_common_color = Counter(pixels).most_common(1)[0][0]
    return closest_color(most_common_color)

# Function to generate HTML with inline CSS
def generate_html_css(description, color):
    prompt = f"""
    Create a single HTML document with inline CSS based on the following details:
    - Element: {description}
    - Color: {color}
    Ensure the CSS styles are embedded directly within the <style> tag of the HTML.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful web development assistant."},
                      {"role": "user", "content": prompt}],
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error with OpenAI API request: {str(e)}")
        return "Error generating HTML"

# Function to hash images for caching
def hash_image(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Cache to avoid duplicate processing
processed_images = {}

# Route for processing image uploads
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = os.path.join('temp', image.filename)
    image.save(image_path)

    image_hash = hash_image(image_path)
    if image_hash in processed_images:
        return jsonify(processed_images[image_hash])

    img = Image.open(image_path)
    description, color = get_description_from_image(img)
    generated_code = generate_html_css(description, color)

    result = {
        'description': description,
        'color': color,
        'generated_code': generated_code
    }
    processed_images[image_hash] = result
    return jsonify(result)

# Extract description and color from the image
def get_description_from_image(image):
    inputs = processor(images=image, text=list(html_element_descriptions.values()), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.matmul(image_features, text_features.T)

    best_match_idx = similarity.argmax().item()
    best_match_label = list(html_element_descriptions.keys())[best_match_idx]
    description = html_element_descriptions[best_match_label]

    dominant_color = get_dominant_color(image)
    return description, dominant_color

# Serve index.html from the root directory
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def serve_favicon():
    return '', 204

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs('temp', exist_ok=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    app.run(host='0.0.0.0', port=5000, debug=True)
