import cv2
import logging
import time
from ultralytics import YOLO
import sys
from ultralytics.nn.autobackend import AutoBackend

# Configure logging
logging.basicConfig(filename="class_counts.log", level=logging.INFO, filemode="w")

# Load a pretrained YOLOv8n model
model = YOLO("models/model_v3_300epochs.pt")

# Define path to video file
source = "sample.mp4"

# Open video capture
cap = cv2.VideoCapture(source)

# Define the desired frame rate (2 frames per second)
frame_rate = 10
previous_time = 0

# Assuming we are only changing the width and maintaining the full height of the image.
# We would need to know the width of the frame to calculate the inner width.
# Since we can't know the width before we actually grab a frame,
# we need to grab one frame from the video to get the dimensions.
ret, frame = cap.read()
if not ret:
    print("Failed to grab a frame from the video.")
    cap.release()
    sys.exit(1)

height, width = frame.shape[:2]

# Calculate x-coordinates for the ROI to maintain the center
# and adjust only the width, e.g., to take 80% of the width
adjusted_width = int(width * 0.4)
x1 = (width - adjusted_width) // 2
x2 = x1 + adjusted_width

# The y-coordinates still span the full height of the image
y1 = 0
y2 = height

# Now we set up the ROI coordinates
roi_coords = (x1, y1, x2, y2)

while cap.isOpened():
    ret, frame = cap.read()
    # print(f"Frame Shape: {frame.shape}")
    if not ret:
        break  # Break the loop if we reach the end of the video

    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time > 1 / frame_rate:
        # Crop the frame to the ROI
        x1, y1, x2, y2 = roi_coords
        roi_frame = frame[y1:y2, x1:x2]
        # cv2.imshow("YOLO with ROI boundary", roi_frame)
        # time.sleep(3)
        # print(f"Jession: {roi_frame.shape}")
        # Run inference on the cropped frame
        results = model.predict(roi_frame, show=True, iou=0.1, line_width=1)

        for result in results:
            print(result.boxes.cls)
        previous_time = current_time

# Release video capture
cap.release()

# Optionally, close any OpenCV windows
cv2.destroyAllWindows()
