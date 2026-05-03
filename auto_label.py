"""
Auto-label cards using template matching.
Uses card_01-card_17 templates to find card positions.
"""
import cv2
import numpy as np
import os
import sys

# Card positions from template matching (17 cards in a hand)
# This maps template names to card positions
CARD_TEMPLATES = {
    'card_01': 0, 'card_02': 1, 'card_03': 2, 'card_04': 3, 'card_05': 4,
    'card_06': 5, 'card_07': 6, 'card_08': 7, 'card_09': 8, 'card_10': 9,
    'card_11': 10, 'card_12': 11, 'card_13': 12, 'card_14': 13, 'card_15': 14,
    'card_16': 15, 'card_17': 16
}

def detect_card_positions(img, template_dir='templates'):
    """Use card_XX templates to find card positions."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    card_positions = []

    for name in [f'card_{i:02d}.png' for i in range(1, 18)]:
        template_path = os.path.join(template_dir, name)
        if not os.path.exists(template_path):
            continue

        template = cv2.imread(template_path)
        if template is None:
            continue

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        if template_gray.shape[0] > h or template_gray.shape[1] > w:
            continue

        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.7:  # Confidence threshold
            x1, y1 = max_loc
            x2, y2 = x1 + template_gray.shape[1], y1 + template_gray.shape[0]
            card_positions.append((x1, y1, x2, y2, max_val, name))

    # NMS
    card_positions = sorted(card_positions, key=lambda x: x[4], reverse=True)
    filtered = []
    while card_positions:
        best = card_positions.pop(0)
        filtered.append(best)
        card_positions = [p for p in card_positions
                          if abs(p[0] - best[0]) > 100 or abs(p[1] - best[1]) > 50]

    filtered.sort(key=lambda x: x[0])  # Sort by x position
    return filtered


def generate_yolo_label(img_path, output_dir, template_dir='templates'):
    """Generate YOLO format label file for an image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f'Failed to load {img_path}')
        return

    h, w = img.shape[:2]
    basename = os.path.splitext(os.path.basename(img_path))[0]

    # Crop hand area (bottom 30%)
    hand_img = img[int(h*0.7):int(h*0.98), :]
    hand_h, hand_w = hand_img.shape[:2]

    # Detect cards in hand area
    positions = detect_card_positions(hand_img, template_dir)

    # Generate YOLO format labels
    # class_id x_center y_center width height (normalized)
    labels = []

    for i, (x1, y1, x2, y2, score, template_name) in enumerate(positions):
        # Convert to YOLO format (normalized coordinates)
        x_center = (x1 + x2) / 2 / hand_w
        y_center = (y1 + y2) / 2 / hand_h
        bw = (x2 - x1) / hand_w
        bh = (y2 - y1) / hand_h

        # Card index tells us approximate position
        # For now, mark all as class 0 (3) - we'll need to identify ranks separately
        # But since we can't identify ranks well, let's use a simpler approach
        # Just assign class based on position index in hand (left to right)
        # This is a placeholder - actual card identification would need more work

        # For training, we just need bounding boxes with class labels
        # Since we can't identify ranks, let's create labels for card positions
        # Class 0 = generic card
        class_id = 0  # Placeholder

        labels.append(f'{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}')

    # Save label file
    label_path = os.path.join(output_dir, f'{basename}.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

    print(f'Generated {label_path} with {len(labels)} cards detected')


def main():
    import glob

    template_dir = 'templates'
    image_dir = 'dataset/images/train'
    label_dir = 'dataset/labels/train'

    os.makedirs(label_dir, exist_ok=True)

    # Process all training images
    images = glob.glob(os.path.join(image_dir, '*.png'))
    images.extend(glob.glob(os.path.join(image_dir, '*.jpg')))

    for img_path in sorted(images):
        print(f'Processing {img_path}')
        generate_yolo_label(img_path, label_dir, template_dir)


if __name__ == '__main__':
    main()
