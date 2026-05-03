"""
Simple card annotation tool using OpenCV.
Draws bounding boxes on cards and saves labels in YOLO format.
"""
import cv2
import os
import numpy as np
from PIL import Image

def pil_imread(path):
    """Read image using PIL which handles Unicode paths better on Windows."""
    img = Image.open(path)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

class SimpleAnnotator:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        os.makedirs(label_dir, exist_ok=True)

        # Get all images
        self.images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.images.extend(sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]))

        if not self.images:
            print(f"No images found in {image_dir}")
            return

        self.current_idx = 0
        self.boxes = []  # Current image's boxes: [(x1, y1, x2, y2), ...]
        self.drawing = False
        self.start_point = None

        # Class definitions (15 classes for card ranks only)
        self.classes = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'Joker_B', 'Joker_R']
        self.class_id = 0  # Current selected class

        print(f"Loaded {len(self.images)} images")
        print("Controls:")
        print("  Left-click + drag: Draw bounding box")
        print("  Left-click: Complete current box and save class")
        print("  z: Undo last box")
        print("  c: Clear all boxes on current image")
        print("  s: Save labels")
        print("  n: Next image")
        print("  p: Previous image")
        print("  0-9, q, w, e, r, t, y: Select class (0-9, q=10, w=11, e=12, r=13, t=14, y=14)")
        print("  ESC: Quit")

    def load_labels(self):
        """Load existing labels for current image if they exist."""
        self.boxes = []
        label_path = os.path.join(self.label_dir, os.path.splitext(self.images[self.current_idx])[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        # Convert from YOLO format to pixel coordinates (original image size)
                        h_img, w_img = self.img.shape[:2]
                        x1 = int((x_center - w/2) * w_img)
                        y1 = int((y_center - h/2) * h_img)
                        x2 = int((x_center + w/2) * w_img)
                        y2 = int((y_center + h/2) * h_img)
                        self.boxes.append((x1, y1, x2, y2, class_id))

    def save_labels(self):
        """Save current boxes to YOLO format label file."""
        if not self.img is None:
            h, w = self.img.shape[:2]
            label_path = os.path.join(self.label_dir, os.path.splitext(self.images[self.current_idx])[0] + '.txt')

            with open(label_path, 'w') as f:
                for box in self.boxes:
                    x1, y1, x2, y2, class_id = box
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n')

            print(f"Saved {len(self.boxes)} boxes to {label_path}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.display = cv2.resize(self.img, (self.scaled_w, self.scaled_h))
                cv2.rectangle(self.display, self.start_point, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1 = min(self.start_point[0], x)
                y1 = min(self.start_point[1], y)
                x2 = max(self.start_point[0], x)
                y2 = max(self.start_point[1], y)
                # Scale back to original image coordinates
                if self.scale < 1.0:
                    x1, y1 = int(x1 / self.scale), int(y1 / self.scale)
                    x2, y2 = int(x2 / self.scale), int(y2 / self.scale)
                # Only add if box is big enough
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.boxes.append((x1, y1, x2, y2, self.class_id))
                    print(f"Added box: class={self.classes[self.class_id]}, ({x1},{y1})-({x2},{y2})")

    def run(self):
        """Main loop."""
        window_name = 'Card Annotator'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.current_idx = 0
        self.img = None
        self.display = None
        self.scale = 1.0
        self.scaled_w = 0
        self.scaled_h = 0

        # Max window size
        self.max_w = 1200
        self.max_h = 800

        while True:
            # Load current image
            if self.img is None:
                img_path = os.path.join(self.image_dir, self.images[self.current_idx])
                self.img = pil_imread(img_path)
                if self.img is None:
                    print(f"Failed to load {img_path}")
                    self.current_idx = (self.current_idx + 1) % len(self.images)
                    continue
                # Calculate scale to fit window
                h, w = self.img.shape[:2]
                self.scale = min(self.max_w / w, self.max_h / h, 1.0)
                self.scaled_w = int(w * self.scale)
                self.scaled_h = int(h * self.scale)
                self.display = cv2.resize(self.img, (self.scaled_w, self.scaled_h))
                self.load_labels()

            # Draw all boxes on display (scaled)
            self.display = cv2.resize(self.img, (self.scaled_w, self.scaled_h))
            for i, box in enumerate(self.boxes):
                x1, y1, x2, y2, class_id = box
                # Scale coordinates
                sx1, sy1 = int(x1 * self.scale), int(y1 * self.scale)
                sx2, sy2 = int(x2 * self.scale), int(y2 * self.scale)
                color = (0, 255, 0) if i == len(self.boxes) - 1 else (255, 0, 0)
                cv2.rectangle(self.display, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(self.display, self.classes[class_id], (sx1, sy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale, (0, 0, 0), 2)

            # Draw class selector on top
            info = f"Image: {self.current_idx+1}/{len(self.images)}"
            cv2.putText(self.display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            class_info = f"Class: {self.class_id} ({self.classes[self.class_id]})"
            cv2.putText(self.display, class_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            box_info = f"Boxes: {len(self.boxes)}"
            cv2.putText(self.display, box_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save_labels()
            elif key == ord('n'):
                self.save_labels()
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self.img = None
            elif key == ord('p'):
                self.save_labels()
                self.current_idx = (self.current_idx - 1) % len(self.images)
                self.img = None
            elif key == ord('z'):
                if self.boxes:
                    self.boxes.pop()
                    print("Undo last box")
            elif key == ord('c'):
                self.boxes = []
                print("Clear all boxes")
            elif key in [ord(str(i)) for i in range(10)]:
                self.class_id = int(chr(key))
                print(f"Selected class: {self.class_id} ({self.classes[self.class_id]})")
            elif key in [ord(c) for c in ['q', 'w', 'e', 'r', 't']]:
                # q=10, w=11, e=12, r=13, t=14
                class_map = {'q': 10, 'w': 11, 'e': 12, 'r': 13, 't': 14}
                self.class_id = class_map[chr(key)]
                print(f"Selected class: {self.class_id} ({self.classes[self.class_id]})")

        self.save_labels()
        cv2.destroyAllWindows()
        print("Annotation session ended")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='dataset/images/train', help='Directory with images')
    parser.add_argument('--label_dir', default='dataset/labels/train', help='Directory for labels')
    args = parser.parse_args()

    annotator = SimpleAnnotator(args.image_dir, args.label_dir)
    annotator.run()