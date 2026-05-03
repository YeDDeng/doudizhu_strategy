"""
对比 YOLO 模型检测结果与实际标注
"""
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'Joker_B', 'Joker_R']

def load_yolo_labels(label_path):
    """加载 YOLO TXT 格式的标签"""
    labels = []
    if not Path(label_path).exists():
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                labels.append({'cls': cls, 'name': CLASS_NAMES[cls], 'cx': cx, 'cy': cy, 'w': w, 'h': h})
    return labels

def detect_with_yolo(img_path, model):
    """使用 YOLO 模型检测"""
    results = model(img_path, conf=0.10, iou=0.4, imgsz=1280, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            detections.append({
                'cls': cls,
                'name': CLASS_NAMES[cls],
                'conf': float(box.conf[0])
            })
    return detections

def analyze_confusion(model_path, test_images, label_dir):
    """分析模型误检模式"""
    from collections import Counter
    model = YOLO(model_path)

    all_false_positives = []
    all_false_negatives = []

    for img_path in test_images:
        img_name = Path(img_path).stem
        label_path = label_dir / f"{img_name}.txt"
        actual_labels = load_yolo_labels(label_path)
        actual_cards = [l['name'] for l in actual_labels]

        yolo_detections = detect_with_yolo(img_path, model)
        yolo_cards = [d['name'] for d in yolo_detections]

        actual_set = set(actual_cards)
        yolo_set = set(yolo_cards)

        # 误检的牌
        false_positives = yolo_set - actual_set
        # 漏检的牌
        false_negatives = actual_set - yolo_set

        all_false_positives.extend(false_positives)
        all_false_negatives.extend(false_negatives)

    print("误检牌统计 (检测到但实际没有):")
    for card, count in Counter(all_false_positives).most_common():
        print(f"  {card}: {count}次")

    print("\n漏检牌统计 (实际有但没检测到):")
    for card, count in Counter(all_false_negatives).most_common():
        print(f"  {card}: {count}次")


def test_with_confidence(model_path, test_images, label_dir, conf_thresholds):
    """测试不同置信度阈值的效果"""
    model = YOLO(model_path)

    print(f"\n{'='*70}")
    print(f"模型: {Path(model_path).name} - 置信度阈值测试")
    print(f"{'='*70}")

    for conf in conf_thresholds:
        total_precision = 0
        total_recall = 0
        image_count = 0

        for img_path in test_images:
            label_path = label_dir / f"{Path(img_path).stem}.txt"
            actual_labels = load_yolo_labels(label_path)
            actual_set = set([l['name'] for l in actual_labels])

            # 用不同置信度检测
            results = model(img_path, conf=conf, iou=0.4, imgsz=1280, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls = int(box.cls[0])
                    detections.append({'cls': cls, 'name': CLASS_NAMES[cls], 'conf': float(box.conf[0])})

            yolo_set = set([d['name'] for d in detections])

            if actual_set:
                correct = len(yolo_set & actual_set)
                precision = correct / len(yolo_set) * 100 if yolo_set else 0
                recall = correct / len(actual_set) * 100 if actual_set else 0
                total_precision += precision
                total_recall += recall
                image_count += 1

        avg_p = total_precision / image_count if image_count else 0
        avg_r = total_recall / image_count if image_count else 0
        print(f"  conf={conf:.2f}: 精确率={avg_p:.1f}% 召回率={avg_r:.1f}%")

if __name__ == '__main__':
    test_images = [
        "C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\images\\database\\微信图片_20260410214738_81_16.png",
        "C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\images\\database\\微信图片_20260410223016_1_5.png",
        "C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\images\\database\\微信图片_20260410223156_5_5.png",
        "C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\images\\database\\微信图片_20260410223605_8_5.png",
        "C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\images\\database\\微信图片_20260415202914_92_16.jpg",
    ]
    label_dir = Path("C:\\Users\\30330\\Desktop\\doudizhu_strategy\\dataset\\labels\\database")

    # 分析 v1 误检模式
    print("\n" + "="*70)
    print("分析 yolov8_cards_v1.pt 误检模式")
    print("="*70)
    analyze_confusion("C:/Users/30330/Desktop/doudizhu_strategy/models/yolov8_cards_v1.pt",
                      test_images, label_dir)

    # 测试 v1 不同置信度阈值
    test_with_confidence("C:/Users/30330/Desktop/doudizhu_strategy/models/yolov8_cards_v1.pt",
                         test_images, label_dir, [0.10, 0.15, 0.20, 0.25, 0.30])

    # 对比 v1 和 new 的置信度
    print("\n" + "="*70)
    print("对比 yolov8_cards_new.pt 不同置信度")
    print("="*70)
    test_with_confidence("C:/Users/30330/Desktop/doudizhu_strategy/models/yolov8_cards_new.pt",
                         test_images, label_dir, [0.05, 0.08, 0.10, 0.15])
