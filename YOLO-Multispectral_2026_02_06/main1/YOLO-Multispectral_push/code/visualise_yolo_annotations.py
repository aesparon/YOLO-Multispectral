
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_yolo_segmentation(images_dir, labels_dir, class_names=None, num_samples=5):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    random.shuffle(image_files)

    for img_file in image_files[:num_samples]:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        if img.ndim == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] > 3:
            img = img[:, :, :3]  # keep only RGB
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Draw polygons
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:  # at least 3 points = 6 coords
                        continue
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * w)
                        y = int(coords[i + 1] * h)
                        points.append((x, y))
                    points = np.array(points, np.int32).reshape((-1, 1, 2))

                    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    label = class_names[cls_id] if class_names else str(cls_id)
                    cv2.putText(img, label, tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            print(f"❌ No label file for: {img_file}")

        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(img_file)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# # ✅ Example usage:
# images_dir = "data/images/train"
# labels_dir = "data/labels/train"
# class_names = ['maize', 'amaranth', 'grass', 'quickweed', 'other']  # optional
# plot_yolo_annotations(images_dir, labels_dir, class_names, num_samples=3)
