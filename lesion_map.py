import cv2
import numpy as np
import os
import uuid

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_lesion_map(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image {img_path}")
    
    # Dummy lesion map (convert to gray for example)
    lesion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lesion = cv2.cvtColor(lesion, cv2.COLOR_GRAY2BGR)

    filename = f"lesion_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, lesion)

    return output_path