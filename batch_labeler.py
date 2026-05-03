"""
批量标注工具：用YOLO检测卡牌位置+类别
直接从模型输出生成YOLO格式标签（无需MiniMax API）

使用方法:
  python batch_labeler.py                    # 标注数据库所有图片
  python batch_labeler.py --dir "path/to/images"  # 指定目录
  python batch_labeler.py --limit 10         # 只处理前10张（测试用）
  python batch_labeler.py --verify            # 验证已标注的图片
"""

import os
import sys
import cv2
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DB_DIR = Path("C:/Users/30330/Desktop/斗地主数据库")
OUTPUT_LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "database"
OUTPUT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "database"


def detect_cards_with_yolo(image, model_path='models/yolov8_cards.pt'):
    """用YOLO检测卡牌位置和类别"""
    from ultralytics import YOLO
    model = YOLO(model_path)

    # 处理灰度图
    if len(image.shape) == 2:
        # 灰度图 -> BGR (3通道)
        processed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # 已经是BGR，先转灰度再处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bright = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)
        # 处理后可能仍是单通道，需要转回BGR
        processed = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)

    results = model(processed, conf=0.10, iou=0.4, imgsz=1280, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'conf': conf,
                'cls': cls,
                'center_x': int((x1 + x2) / 2),
                'center_y': int((y1 + y2) / 2)
            })

    detections.sort(key=lambda d: d['center_x'])
    return detections


def label_image(img_path, model_path='models/yolov8_cards.pt', output_dir=None, copy_image_dir=None):
    """对单张图片进行YOLO标注（直接使用模型输出的类别）"""
    img = cv2.imread(img_path)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]

    h, w = img.shape[:2]
    img_basename = os.path.basename(img_path)
    print(f"\n处理: {img_basename} ({w}x{h})")

    # YOLO检测
    yolo_dets = detect_cards_with_yolo(img, model_path)
    print(f"  YOLO检测到 {len(yolo_dets)} 个候选")

    if not yolo_dets:
        print("  [SKIP] YOLO未检测到任何卡牌")
        return False

    # 生成标签（过滤只保留底部区域 y >= 0.5）
    final_labels = []
    bottom_count = 0
    for det in yolo_dets:
        cy_norm = det['center_y'] / h
        # 只保留底部区域 (y >= 50%)
        if cy_norm >= 0.5:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2 / w
            cy = cy_norm
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            final_labels.append(f"{det['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            bottom_count += 1

    print(f"  底部区域保留 {bottom_count} 个标签 (过滤掉顶部)")

    # 保存标签
    if output_dir is None:
        output_dir = OUTPUT_LABEL_DIR
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')

    with open(label_path, 'w') as f:
        f.write('\n'.join(final_labels))
    print(f"  [OK] 已保存标签: {len(final_labels)} 张牌")

    # 复制图片
    if copy_image_dir is not None:
        os.makedirs(copy_image_dir, exist_ok=True)
        import shutil
        dest_img = os.path.join(copy_image_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest_img)

    return True


def main():
    parser = argparse.ArgumentParser(description='批量YOLO标注')
    parser.add_argument('--dir', '-d', type=str, default=str(DB_DIR), help='图片目录')
    parser.add_argument('--limit', '-l', type=int, default=0, help='处理数量限制(0=全部)')
    parser.add_argument('--model', '-m', type=str, default='models/yolov8_cards.pt', help='YOLO模型路径')
    parser.add_argument('--skip-labeled', action='store_true', help='跳过已有标签的图片')
    args = parser.parse_args()

    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.dir)
    files = sorted(os.listdir(img_dir))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(image_files)} 张图片")

    # 过滤已标注
    if args.skip_labeled:
        unlabeled = []
        for fname in image_files:
            label_path = OUTPUT_LABEL_DIR / (Path(fname).stem + '.txt')
            if not label_path.exists():
                unlabeled.append(fname)
        print(f"未标注: {len(unlabeled)} 张")
        image_files = unlabeled

    if args.limit > 0:
        image_files = image_files[:args.limit]
        print(f"限制处理: {len(image_files)} 张")

    print(f"\n开始处理 {len(image_files)} 张图片...")
    print("=" * 60)

    success = 0
    failed = 0
    results = []

    for i, fname in enumerate(image_files):
        img_path = img_dir / fname
        print(f"\n[{i+1}/{len(image_files)}]")

        try:
            label_path = OUTPUT_LABEL_DIR / (Path(fname).stem + '.txt')
            if args.skip_labeled and label_path.exists():
                print(f"  [SKIP] 已有标签")
                continue

            if label_image(str(img_path), args.model, str(OUTPUT_LABEL_DIR), str(OUTPUT_IMAGE_DIR)):
                success += 1
                results.append({'file': fname, 'status': 'ok'})
            else:
                failed += 1
                results.append({'file': fname, 'status': 'failed'})

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            results.append({'file': fname, 'status': 'error', 'error': str(e)})

    # 保存报告
    report = {
        'time': datetime.now().isoformat(),
        'total': len(image_files),
        'success': success,
        'failed': failed,
        'results': results
    }

    report_path = PROJECT_ROOT / "dataset" / "labels" / "database" / "labeling_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"完成: 成功{success}/{len(image_files)} 张")
    print(f"报告已保存: {report_path}")


if __name__ == '__main__':
    import numpy as np
    main()