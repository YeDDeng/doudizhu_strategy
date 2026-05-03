"""
Simple test script for card detection.
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.card_recognizer import CardRecognizer
import cv2

def main():
    print("=" * 50)
    print("Testing Card Detection")
    print("=" * 50)

    # Initialize recognizer
    recognizer = CardRecognizer(
        model_path="models/yolov8_cards.pt",
        confidence_threshold=0.25
    )

    # Test images
    test_images = [
        "dataset/images/test_81_16.png",
        "dataset/images/test_120_16.png",
    ]

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\nImage not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"\nFailed to load: {img_path}")
            continue

        detections = recognizer.detect_cards(img)
        print(f"\n{img_path}:")
        print(f"  Detected {len(detections)} cards:")
        for d in detections:
            print(f"    {d['card']}: conf={d['confidence']:.3f}")

if __name__ == "__main__":
    main()