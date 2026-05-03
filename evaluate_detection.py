"""
用已有标注数据评估YOLO检测效果
对比模型检测结果 vs 标注(ground truth)
"""
import cv2
import os
import sys
import numpy as np
from collections import Counter, defaultdict

def imread_unicode(path):
    """解决OpenCV在Windows下无法读取中文路径的问题"""
    try:
        import numpy as np
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.card_recognizer import CardRecognizer

# Class ID to rank mapping (与dataset.yaml一致)
CLASS_MAP = {
    0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8',
    6: '9', 7: '10', 8: 'J', 9: 'Q', 10: 'K', 11: 'A',
    12: '2', 13: 'Joker_B', 14: 'Joker_R'
}

def parse_yolo_label(label_path, img_w, img_h):
    """解析YOLO格式标注文件，返回卡牌列表(按x位置排序)"""
    cards = []
    if not os.path.exists(label_path):
        return cards
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            x1 = int(x_center - bw/2)
            y1 = int(y_center - bh/2)
            x2 = int(x_center + bw/2)
            y2 = int(y_center + bh/2)
            rank = CLASS_MAP.get(cls_id, '?')
            cards.append({
                'rank': rank,
                'class_id': cls_id,
                'bbox': (x1, y1, x2, y2),
                'center_x': int(x_center),
                'center_y': int(y_center)
            })
    # 按x位置排序
    cards.sort(key=lambda c: c['center_x'])
    return cards

def yolo_to_card_list(detections):
    """把YOLO检测结果转成rank列表(按center_x排序)"""
    cards = []
    for d in detections:
        cards.append({
            'rank': d['card'],
            'center_x': d['center_x'],
            'bbox': d['bbox']
        })
    cards.sort(key=lambda c: c['center_x'])
    return cards

def compute_metrics(pred_list, gt_list):
    """
    计算检测指标
    pred_list / gt_list: 按x位置排序的card列表
    使用贪心匹配：按位置顺序匹配，容许x位置误差<=50px
    """
    matched = 0
    false_positives = 0
    false_negatives = 0

    gt_used = [False] * len(gt_list)
    pred_used = [False] * len(pred_list)

    # 贪心匹配
    for p in range(len(pred_list)):
        best_j = -1
        best_dist = float('inf')
        for g in range(len(gt_list)):
            if gt_used[g]:
                continue
            # 匹配条件：rank相同 且 x位置差距<=50px
            if pred_list[p]['rank'] != gt_list[g]['rank']:
                continue
            dist = abs(pred_list[p]['center_x'] - gt_list[g]['center_x'])
            if dist <= 50 and dist < best_dist:
                best_dist = dist
                best_j = g
        if best_j >= 0:
            matched += 1
            gt_used[best_j] = True
            pred_used[p] = True
        else:
            false_positives += 1

    for g in range(len(gt_list)):
        if not gt_used[g]:
            false_negatives += 1

    precision = matched / (matched + false_positives) if (matched + false_positives) > 0 else 0
    recall = matched / (matched + false_negatives) if (matched + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'matched': matched,
        'fp': false_positives,
        'fn': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    dataset_dir = 'dataset'
    images_dir = os.path.join(dataset_dir, 'images', 'all')
    labels_dir = os.path.join(dataset_dir, 'labels', 'all')

    # 加载模型
    recognizer = CardRecognizer(
        model_path='models/yolov8_cards.pt',
        confidence_threshold=0.15
    )

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))])

    total_metrics = {
        'matched': 0, 'fp': 0, 'fn': 0,
        'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    }

    error_cases = []

    print(f"Testing on {len(image_files)} images...")
    print("=" * 70)

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        img = imread_unicode(img_path)
        if img is None:
            print(f"[SKIP] {img_file}: cannot load image")
            continue

        h, w = img.shape[:2]

        # Ground truth
        gt_cards = parse_yolo_label(label_path, w, h)
        if not gt_cards:
            print(f"[SKIP] {img_file}: no labels")
            continue

        # YOLO检测：截取底部手牌区域 (0.05, 0.60, 0.95, 0.95)
        my_hand_roi = img[int(0.60*h):int(0.95*h), int(0.05*w):int(0.95*w)]
        detections = recognizer.detect_cards(my_hand_roi, card_type='my')
        pred_cards = yolo_to_card_list(detections)

        # 转换为相对于原图的center_x
        offset_x = int(0.05 * w)
        for c in pred_cards:
            c['center_x'] += offset_x

        # 计算指标
        metrics = compute_metrics(pred_cards, gt_cards)

        # 统计每类错误
        gt_ranks = Counter(c['rank'] for c in gt_cards)
        pred_ranks = Counter(c['rank'] for c in pred_cards)

        for rank, count in gt_ranks.items():
            total_metrics['per_class'][rank]['tp'] += min(count, pred_ranks.get(rank, 0))
            total_metrics['per_class'][rank]['fn'] += max(0, count - pred_ranks.get(rank, 0))
        for rank, count in pred_ranks.items():
            total_metrics['per_class'][rank]['fp'] += max(0, count - gt_ranks.get(rank, 0))

        total_metrics['matched'] += metrics['matched']
        total_metrics['fp'] += metrics['fp']
        total_metrics['fn'] += metrics['fn']

        # 记录错误案例
        if metrics['f1'] < 0.8:
            error_cases.append({
                'file': img_file,
                'gt': [c['rank'] for c in gt_cards],
                'pred': [c['rank'] for c in pred_cards],
                'metrics': metrics
            })

        status = "[OK]" if metrics['f1'] >= 0.95 else "[WARN]" if metrics['f1'] >= 0.8 else "[FAIL]"
        print(f"{status} {img_file}: GT={len(gt_cards)}, PRED={len(pred_cards)}, "
              f"M={metrics['matched']}, FP={metrics['fp']}, FN={metrics['fn']}, "
              f"P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")

    print("=" * 70)
    print("\n=== OVERALL RESULTS ===")
    total_pred = total_metrics['matched'] + total_metrics['fp']
    total_gt = total_metrics['matched'] + total_metrics['fn']
    overall_p = total_metrics['matched'] / total_pred if total_pred > 0 else 0
    overall_r = total_metrics['matched'] / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    print(f"Total GT cards: {total_gt}, Total predicted: {total_pred}")
    print(f"Overall: Precision={overall_p:.3f}, Recall={overall_r:.3f}, F1={overall_f1:.3f}")
    print(f"Matched={total_metrics['matched']}, FP={total_metrics['fp']}, FN={total_metrics['fn']}")

    print("\n=== PER-CLASS PERFORMANCE ===")
    for rank in sorted(total_metrics['per_class'].keys()):
        stats = total_metrics['per_class'][rank]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {rank:>6}: TP={tp:3d}, FP={fp:3d}, FN={fn:3d} | P={p:.2f}, R={r:.2f}, F1={f:.2f}")

    if error_cases:
        print(f"\n=== ERROR CASES ({len(error_cases)}) ===")
        for case in error_cases[:5]:
            print(f"  {case['file']}: GT={case['gt']} → PRED={case['pred']} (F1={case['metrics']['f1']:.2f})")

if __name__ == '__main__':
    main()
