from ultralytics import YOLO
import cv2

# Load a model
model = YOLO(r"models\model_v2.pt")  # pretrained YOLOv8n model


video_path = r"Datasets\videos\cbf0a4bf-7033-466d-acfd-99b70aa30994 - Trim.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(
            frame,
            # source=r"Datasets\videos\cbf0a4bf-7033-466d-acfd-99b70aa30994 - Trim.mp4",
            # stream=True,
            conf=0.2,
            # show=True,
            # # save=True,
            # # save_txt=True,
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

# # Process results generator
# for result in results:
#     # print(result)
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
