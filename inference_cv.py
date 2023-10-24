from ultralytics import YOLO
import cv2

# Load a model
model = YOLO(r"models/model_v2.pt")  # pretrained YOLOv8n model

video_path = r"sample.mp4"
cap = cv2.VideoCapture(video_path)

# Retrieve the original frame rate of the video
original_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate how many frames to skip to process at 3
frame_skip = original_fps // 5

# Initialize a frame counter
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1  # Increment the frame counter
    if success:
        # Only process every frame_skip frames
        if frame_count % frame_skip == 0:
            results = model(
                frame,
                conf=0.12,
            )  # return a generator of Results objects
            annotated_frame = results[0].plot()

            # Define the new width and height
            new_width = int(annotated_frame.shape[1] * 0.6)  # reducing width by 20%
            new_height = int(annotated_frame.shape[0] * 0.45)  # reducing height by 20%

            # Resize the frame
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            cv2.imshow("Yolov8 inference", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()
