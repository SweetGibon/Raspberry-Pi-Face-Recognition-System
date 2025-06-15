# Raspberry Pi Real-Time Face Recognition System

This project is a real-time face recognition system using a Raspberry Pi, camera module, and ONNX-based facial recognition model. It captures a live image using the Pi Camera, compares it against a pre-trained dataset, and identifies the person.

# What It Does

- Takes a photo using the Raspberry Pi camera module.
- Runs facial recognition using embeddings generated from a pre-trained model.
- Identifies the person if they match with known dataset faces.
- Displays the captured image and prints the result to the terminal.

# Key Features

- Uses 'onnxruntime' with a pre-trained InsightFace model.
- Supports multiple known users via dataset folder structure.
- Modular code split into training, recognition, and capture scripts.
- Designed for performance on Raspberry Pi with camera module 2.

# Project Structure

```bash
face_recognition_pi/
│
├── train.py              # Generates embeddings from dataset images
├── recognizer.py         # Captures image and identifies person
├── capture_and_show.py   # Captures image using the Raspberry Pi camera
├── embeddings/           # Stores .pkl face embeddings for each known person
├── Dataset/              # Organized dataset of known people (images not included in repo)
├── models/               # ONNX model directory (excluded from repo, see below)

