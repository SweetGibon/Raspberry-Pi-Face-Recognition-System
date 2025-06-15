import onnxruntime as ort
import numpy as np
import os
import cv2
import pickle
import time
from picamera2 import Picamera2

# Initialize ONNX model
model_path = '<path to the w600k_r50.onnx model>'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Preprocess image
def preprocess_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 127.5) - 1.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# Get embedding from image
def get_embedding(image_path):
    face = preprocess_face(image_path)
    embedding = session.run(None, {input_name: face})[0]
    embedding = embedding.flatten()
    return normalize_embedding(embedding)

# Normalize embedding
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm

# Load known embeddings
def load_known_embeddings(embeddings_dir='embeddings'):
    known = {}
    for person in os.listdir(embeddings_dir):
        person_folder = os.path.join(embeddings_dir, person)
        if os.path.isdir(person_folder):
            emb_list = []
            for file in os.listdir(person_folder):
                if file.endswith('.pkl'):
                    with open(os.path.join(person_folder, file), 'rb') as f:
                        emb = pickle.load(f)
                        emb_list.append(normalize_embedding(emb))
            known[person] = emb_list
    return known

# Compare captured face with known faces
def recognize_face(image_path, threshold=1.1):
    target_emb = get_embedding(image_path)
    known_embeddings = load_known_embeddings()

    identity = "Unknown"
    min_dist = float('inf')

    for person, embeddings in known_embeddings.items():
        for emb in embeddings:
            dist = np.linalg.norm(emb - target_emb)
            print(f"Comparing to {person} → Distance: {dist:.4f}")
            if dist < threshold and dist < min_dist:
                min_dist = dist
                identity = person
    return identity

# Capture photo from PiCamera and return path
def capture_image(save_dir='capturedimage', filename='live_capture.jpg'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(2)  # warm-up
    img = picam2.capture_array()

    cv2.imwrite(filepath, img)
    print(f"Image captured and saved to {filepath}")
    return filepath

# Main logic
if __name__ == "__main__":
    img_path = capture_image()
    identity = recognize_face(img_path, threshold=1.0)
    print(f"Identified as: {identity}")
