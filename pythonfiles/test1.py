import os
from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("best.pt")
class_names = "fault"

# Directory containing images
images_dir = r"C:\Users\Jeyanth\Downloads\pothole_dataset_v8\pothole_dataset_v8\train\images"

# List all images in the directory
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    img = cv2.imread(os.path.join(images_dir, image_file))
    img = cv2.resize(img, (1020, 500))  # Resize image if needed
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  
        masks = r.masks  

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to close the window
        break

cv2.destroyAllWindows()