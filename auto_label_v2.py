"""
半自动标注工具：用YOLO检测卡牌位置 + MiniMax识别牌面
生成YOLO格式标签文件

使用方法:
  python auto_label_v2.py                    # 标注train目录所有图片
  python auto_label_v2.py --file xxx.png       # 标注单个文件
  python auto_label_v2.py --verify            # 仅验证已标注的图片
"""
import os
import sys
import cv2
import numpy as np
import time
from collections import Counter

# MiniMax MCP
from mcp__MiniMax__understand_image import understand_image

# 牌面值映射 (与dataset.yaml一致)
CLASS_NAME_TO_ID = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5,
    '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,
    '2': 12, '小王': 13, '大王': 14, 'Joker_B': 13, 'Joker_R': 14,
    'joker_b': 13, 'joker_r': 14, 'black_joker': 13, 'red_joker': 14,
    '小王(黑桃)': 13, '大王(红桃)': 14
}

# 归一化YOLO格式: class_id x_center y_center width height
def parse_minimax_cards(text):
    """
    解析MiniMax返回的文本，提取卡牌列表
    返回: list of (rank_str, suit_str) 或 None
    """
    import re
    cards = []

    # 提取花色符号
    suits = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C', 'spade': 'S', 'heart': 'H', 'diamond': 'D', 'club': 'C'}

    # 提取所有"XX花色"或"花色XX"的模式
    # 例如: "A♥", "10♠", "小王", "大王"
    patterns = [
        r'([3-9]|10|J|Q|K|A|2|小王|大王|Joker_B|Joker_R|黑王|红王)([♠♥♦♣])',  # A♥ 格式
        r'([♠♥♦♣])([3-9]|10|J|Q|K|A|2|小王|大王)',  # ♥A 格式
        r'(小王|大王|黑王|红王|Joker_B|Joker_R)',  # 单独的王
        r'([3-9]|10|J|Q|K|A|2)(spade|heart|diamond|club|黑桃|红桃|方块|梅花)',  # A spade 格式
    ]

    found = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for m in matches:
            rank = m.group(1)
            suit = m.group(2) if len(m.groups()) > 1 else ''

            # 标准化rank
            rank_map = {
                '小王': 'Joker_B', '大王': 'Joker_R',
                '黑王': 'Joker_B', '红王': 'Joker_R',
                'Joker_B': 'Joker_B', 'Joker_R': 'Joker_R',
                'spade': 'S', 'heart': 'H', 'diamond': 'D', 'club': 'C',
                '黑桃': 'S', '红桃': 'H', '方块': 'D', '梅花': 'C'
            }
            suit_map = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}

            r = rank_map.get(rank, rank)
            s = suit_map.get(suit, suit) if suit else ''

            if r in ('Joker_B', 'Joker_R'):
                found.append((r, ''))
            elif r in ('3','4','5','6','7','8','9','10','J','Q','K','A','2'):
                found.append((r, s))

    return found


def detect_cards_with_yolo(image, model_path='models/yolov8_cards.pt'):
    """用YOLO检测卡牌位置"""
    from ultralytics import YOLO
    model = YOLO(model_path)

    # CLAHE预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bright = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)
    processed = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)

    results = model(processed, conf=0.10, iou=0.4, imgsz=640, verbose=False)

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

    # 按x位置排序
    detections.sort(key=lambda d: d['center_x'])
    return detections


def label_image(img_path, model_path='models/yolov8_cards.pt', output_dir=None):
    """
    对单张图片进行标注
    流程: YOLO检测位置 -> MiniMax识别牌面 -> 合并生成标签
    """
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        # 尝试用PIL读取中文路径
        from PIL import Image
        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

    h, w = img.shape[:2]
    img_basename = os.path.basename(img_path)
    print(f"\n处理: {img_basename} ({w}x{h})")

    # Step 1: YOLO检测卡牌位置
    yolo_dets = detect_cards_with_yolo(img, model_path)
    print(f"  YOLO检测到 {len(yolo_dets)} 个候选")

    if not yolo_dets:
        print("  [SKIP] YOLO未检测到任何卡牌")
        return False

    # Step 2: MiniMax识别牌面
    # 将图片转为base64用于传输
    import base64
    import io
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    # 调用MiniMax识别
    prompt = (
        "这是一张斗地主游戏截图。请识别底部的手牌区域（屏幕下方约60%-95%的区域）中所有卡牌。"
        "按从左到右顺序列出每张牌，例如：A♥、10♠、小王、大王、3♣等。"
        "如果没有看到手牌区域，也请说明看到了什么区域。"
    )

    try:
        result = understand_image(image_source=img_b64, prompt=prompt)
        minimax_text = result.get('text', '') if isinstance(result, dict) else str(result)
        print(f"  MiniMax返回: {minimax_text[:100]}...")
    except Exception as e:
        print(f"  MiniMax调用失败: {e}")
        return False

    # Step 3: 解析MiniMax结果
    minimax_cards = parse_minimax_cards(minimax_text)
    print(f"  MiniMax识别到 {len(minimax_cards)} 张牌: {[c[0]+c[1] for c in minimax_cards]}")

    if not minimax_cards:
        print("  [WARN] MiniMax未识别到任何牌，尝试用YOLO结果")
        # 用YOLO的class作为fallback
        final_labels = []
        for det in yolo_dets:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            final_labels.append(f"{det['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    else:
        # Step 4: 合并YOLO位置 + MiniMax牌面
        # 策略: YOLO提供bbox位置，MiniMax提供class
        # 假设YOLO检测数量和MiniMax识别数量基本一致，按x位置顺序匹配

        final_labels = []
        yolo_classes = [d['cls'] for d in yolo_dets]
        yolo_bboxes = [d['bbox'] for d in yolo_dets]

        # 如果数量不一致，尝试智能匹配
        if len(yolo_dets) == len(minimax_cards):
            # 完美匹配: 1:1对应
            for i, (det, (rank, suit)) in enumerate(zip(yolo_dets, minimax_cards)):
                cls_id = CLASS_NAME_TO_ID.get(rank, 0)
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                final_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                print(f"  [{i}] YOLO:cls={yolo_classes[i]} -> MiniMax:{rank}{suit} | conf={det['conf']:.2f}")
        else:
            # 数量不一致: 尝试按rank频率匹配
            print(f"  [WARN] 数量不匹配 YOLO={len(yolo_dets)} MiniMax={len(minimax_cards)}")
            print(f"  用YOLO位置 + MiniMax rank匹配")

            # 统计YOLO各类数量
            yolo_counter = Counter(yolo_classes)
            mm_counter = Counter([r for r, s in minimax_cards])

            # 假设YOLO的class顺序(按x)和MiniMax的rank顺序一致
            # 用YOLO的bbox位置，但class用MiniMax的
            # 按x位置从左到右，分配MiniMax的ranks

            # 取两者的较小数量进行贪心匹配
            n_match = min(len(yolo_dets), len(minimax_cards))

            # 按x排序后的索引
            yolo_sorted = sorted(enumerate(yolo_dets), key=lambda x: x[1]['center_x'])
            minimax_ranks = [r for r, s in minimax_cards]

            for i in range(n_match):
                det_idx, det = yolo_sorted[i]
                rank = minimax_ranks[i]
                cls_id = CLASS_NAME_TO_ID.get(rank, 0)
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                final_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                print(f"  [{i}] {rank}: YOLO_cls={yolo_classes[det_idx]} -> MiniMax:{rank} | conf={det['conf']:.2f}")

    # Step 5: 保存标签文件
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(img_path), '..', 'labels', 'train')

    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')

    with open(label_path, 'w') as f:
        f.write('\n'.join(final_labels))

    print(f"  [OK] 已保存: {label_path} ({len(final_labels)} 张牌)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='半自动标注: YOLO + MiniMax')
    parser.add_argument('--file', '-f', type=str, help='标注单个文件')
    parser.add_argument('--verify', '-v', action='store_true', help='验证已标注的图片')
    parser.add_argument('--model', '-m', type=str, default='models/yolov8_cards.pt', help='YOLO模型路径')
    args = parser.parse_args()

    if args.file:
        label_image(args.file, args.model)
    elif args.verify:
        # 验证模式: 对比YOLO检测和已保存的标签
        print("验证模式: 对比YOLO检测 vs 已保存标签")
    else:
        # 批量标注train目录
        train_dir = 'dataset/images/train'
        files = sorted(os.listdir(train_dir))
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"找到 {len(image_files)} 张图片待标注")

        success = 0
        for i, fname in enumerate(image_files):
            img_path = os.path.join(train_dir, fname)
            print(f"\n[{i+1}/{len(image_files)}]")
            try:
                if label_image(img_path, args.model):
                    success += 1
                    time.sleep(1)  # 避免API限流
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(3)

        print(f"\n\n=== 完成: {success}/{len(image_files)} 张标注成功 ===")


if __name__ == '__main__':
    main()
