import cv2
import logging
import time
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

# Configure logging
logging.basicConfig(filename="class_counts.log", level=logging.INFO, filemode="w")

# Load a pretrained YOLOv8n model
model = YOLO("models/model_v3_300epochs.pt")

# Define path to video file
source = "data_v1.mp4"

# Open video capture
cap = cv2.VideoCapture(source)

# Define the desired frame rate (2 frames per second)
frame_rate = 2
previous_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if we reach the end of the video

    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time > 1 / frame_rate:
        # Run inference on the current frame
        results = model.predict(frame, show=True, imgsz=640, iou=0.1, line_width=1)

        # ... rest of your processing code ...
        for result in results:
            print(result.boxes.cls)
        previous_time = current_time

# Release video capture
cap.release()

# Optionally, close any OpenCV windows
cv2.destroyAllWindows()
