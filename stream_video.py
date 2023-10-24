from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("models/model_v1.pt")

# Define path to video file
source = "sample.mp4"

# Run inference on the source
results = model(source, stream=True, conf=0.2, save_conf=True, show=True)

for result in results:
    print(result)
    # result.boxes
