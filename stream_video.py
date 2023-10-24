from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("models/model_large_v1.pt")

# Define path to video file
source = "sample.mp4"

# Run inference on the source
results = model(source, stream=True, conf=0.12, show=True)

# Thresholds for each class
thresholds = {0: 0.18, 2: 0.70, 1: 0.7}
count = 0

for result in results:
    if count == 0:
        class_names = result.names
        count += 1
    classes = result.boxes.cls
    confidences = result.boxes.conf
    # Step 1: Map classes to confidences
    class_confidence_pairs = list(zip(classes.numpy(), confidences.numpy()))
    # Step 2: Filter out pairs based on the threshold for each class
    filtered_pairs = [
        pair for pair in class_confidence_pairs if pair[1] >= thresholds.get(pair[0], 0)
    ]
    # Step 3: Count occurrences of each class in the filtered list
    class_counts = {}
    for pair in filtered_pairs:
        class_value = int(pair[0])
        class_counts[class_value] = class_counts.get(class_value, 0) + 1

    print(f"Class Counts: {class_counts}")
