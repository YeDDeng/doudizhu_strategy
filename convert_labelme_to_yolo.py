"""
将 labelme JSON 标注转换为 YOLO TXT 格式
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# 类别映射
CLASS_NAMES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'Joker_B', 'Joker_R']
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def convert_json_to_yolo(json_path: str, output_dir: str = None) -> List[str]:
    """将单个 labelme JSON 文件转换为 YOLO TXT 格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']
    shapes = data['shapes']

    yolo_lines = []
    for shape in shapes:
        label = shape['label']
        if label not in LABEL_TO_ID:
            print(f"  警告: 未知标签 '{label}'，跳过")
            continue

        shape_type = shape.get('shape_type', 'rectangle')

        # 只处理矩形框，跳过其他类型（point, polygon等）
        if shape_type != 'rectangle':
            continue

        class_id = LABEL_TO_ID[label]
        points = shape['points']  # [[x1, y1], [x2, y2]]

        if len(points) < 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        # 计算 YOLO 格式 (归一化的中心点 + 宽高)
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = abs(x2 - x1) / image_width
        height = abs(y2 - y1) / image_height

        # 限制在 0-1 范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存 TXT 文件
    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    txt_filename = os.path.splitext(os.path.basename(json_path))[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))

    return yolo_lines


def main():
    import argparse
    parser = argparse.ArgumentParser(description='转换 labelme JSON 为 YOLO TXT')
    parser.add_argument('--input', '-i', type=str,
                       default='C:/Users/30330/Desktop/doudizhu_strategy/dataset/labels/database',
                       help='JSON 文件目录')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录（默认与输入相同）')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    json_files = list(input_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    success = 0
    failed = 0

    for json_file in json_files:
        try:
            lines = convert_json_to_yolo(str(json_file), str(output_dir))
            success += 1
            print(f"  转换: {json_file.name} -> {len(lines)} 个标注")
        except Exception as e:
            print(f"  错误: {json_file.name} - {e}")
            failed += 1

    print(f"\n完成: 成功 {success}, 失败 {failed}")


if __name__ == '__main__':
    main()
