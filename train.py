import onnxruntime as ort
import numpy as np
import cv2
import os
import pickle

# Load ONNX model (update this path for your system)
model_path = "<Path to the w600k_r50.onnx model>"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name


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


def get_embedding(image_path):
    face = preprocess_face(image_path)
    embedding = session.run(None, {input_name: face})[0]
    return embedding.flatten()


def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def save_all_embeddings(dataset_root='Dataset', embeddings_folder='embeddings'):
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    for person_name in os.listdir(dataset_root):
        person_folder = os.path.join(dataset_root, person_name)
        if os.path.isdir(person_folder):
            print(f"Processing {person_name}...")
            idx = 1
            for img_file in os.listdir(person_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, img_file)
                    try:
                        emb = get_embedding(img_path)
                        emb = normalize_embedding(emb)
                        person_embedding_folder = os.path.join(embeddings_folder, person_name)
                        os.makedirs(person_embedding_folder, exist_ok=True)
                        save_path = os.path.join(person_embedding_folder, f"{idx}.pkl")

                        with open(save_path, 'wb') as f:
                            pickle.dump(emb, f)
                        print(f"Saved embedding for {img_path} as {save_path}")
                        idx += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    save_all_embeddings()
