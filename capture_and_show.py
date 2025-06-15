import time
import cv2
from picamera2 import Picamera2
import os
from datetime import datetime

def main():
    # Define the directory where you want to save images
    save_dir = "<Directory Path>"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())

    # Start camera
    print("Starting camera...")
    picam2.start()
    time.sleep(2)  # Allow camera to adjust

    # Capture image
    print("Capturing image...")
    image = picam2.capture_array()

    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_image_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    # Save image
    cv2.imwrite(filepath, image)
    print(f" Photo saved as {filepath}")

    # Stop camera
    picam2.stop()
    print("Camera stopped.")

if __name__ == "__main__":
    main()
