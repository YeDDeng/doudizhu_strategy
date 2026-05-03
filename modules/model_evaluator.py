"""
模型评估器 - 对比新旧模型精度

使用方法:
  python -c "from modules.model_evaluator import ModelEvaluator; e = ModelEvaluator(); result = e.evaluate('models/v5.pt', 'models/v4.pt')"
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from collections import Counter

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


CLASS_NAMES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'Joker_B', 'Joker_R']


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, test_images_dir: str = None):
        self.project_root = Path(__file__).parent.parent
        self.test_images_dir = test_images_dir or str(self.project_root / "dataset" / "images" / "train")

    def evaluate_detection(self, model_path: str, test_dir: str = None) -> Dict:
        """
        评估单个模型的检测效果

        Returns:
            {
                'total_detections': int,
                'avg_confidence': float,
                'per_class': {class_name: {'count': int, 'avg_conf': float}},
                'low_conf_classes': [class_names with low avg conf]
            }
        """
        if not ULTRALYTICS_AVAILABLE:
            return {'error': 'ultralytics not available'}

        test_dir = test_dir or self.test_images_dir
        model = YOLO(model_path)

        all_detections = []
        per_class_stats = {c: [] for c in CLASS_NAMES}

        import os
        image_files = []
        for f in os.listdir(test_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(test_dir, f))

        for img_path in image_files[:50]:  # 最多测试50张
            try:
                # CLAHE预处理
                import cv2
                img = cv2.imread(img_path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                bright = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)
                processed = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)

                results = model(processed, conf=0.05, iou=0.4, imgsz=1280, verbose=False)

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

                        all_detections.append({'class': class_name, 'conf': conf})
                        per_class_stats[class_name].append(conf)

            except Exception as e:
                continue

        # 计算统计
        avg_conf = sum(d['conf'] for d in all_detections) / len(all_detections) if all_detections else 0

        per_class_result = {}
        low_conf_classes = []
        for cname, confs in per_class_stats.items():
            if confs:
                avg = sum(confs) / len(confs)
                per_class_result[cname] = {'count': len(confs), 'avg_conf': avg}
                if avg < 0.3:
                    low_conf_classes.append(cname)
            else:
                per_class_result[cname] = {'count': 0, 'avg_conf': 0}

        return {
            'total_detections': len(all_detections),
            'avg_confidence': round(avg_conf, 4),
            'per_class': per_class_result,
            'low_conf_classes': low_conf_classes,
            'tested_images': min(50, len(image_files))
        }

    def compare_models(self, new_model: str, old_model: str) -> Dict:
        """
        对比两个模型的检测效果

        Returns:
            {
                'new_model': {...detection_stats},
                'old_model': {...detection_stats},
                'is_better': bool,
                'improvement': {'overall': float, 'per_class': {...}}
            }
        """
        print(f"对比模型: {new_model} vs {old_model}")

        new_stats = self.evaluate_detection(new_model)
        old_stats = self.evaluate_detection(old_model)

        # 计算改进
        new_avg = new_stats.get('avg_confidence', 0)
        old_avg = old_stats.get('avg_confidence', 0)
        overall_improvement = new_avg - old_avg

        # 每类别改进
        per_class_improvement = {}
        for cname in CLASS_NAMES:
            new_conf = new_stats['per_class'].get(cname, {}).get('avg_conf', 0)
            old_conf = old_stats['per_class'].get(cname, {}).get('avg_conf', 0)
            per_class_improvement[cname] = {
                'new': round(new_conf, 4),
                'old': round(old_conf, 4),
                'diff': round(new_conf - old_conf, 4)
            }

        # 新模型是否更好：整体置信度提升且没有类别退化超过10%
        is_better = overall_improvement > 0.01  # 至少提升1%

        # 检查是否有类别严重退化
        severe_regression = False
        for cname, stats in per_class_improvement.items():
            if stats['old'] > 0.1 and stats['diff'] < -0.15:  # 旧置信度>0.1但退化>15%
                severe_regression = True
                break

        if severe_regression:
            is_better = False

        return {
            'new_model': new_stats,
            'old_model': old_stats,
            'is_better': is_better,
            'improvement': {
                'overall': round(overall_improvement, 4),
                'per_class': per_class_improvement
            }
        }

    def should_update_best(self, new_model: str, current_best: str) -> Tuple[bool, str]:
        """
        判断是否应该更新最佳模型

        Returns:
            (should_update, reason)
        """
        if not Path(new_model).exists():
            return False, f"新模型不存在: {new_model}"

        if not Path(current_best).exists():
            return True, "当前最佳不存在，直接使用新模型"

        result = self.compare_models(new_model, current_best)

        if result['is_better']:
            improvement = result['improvement']['overall']
            return True, f"新模型更好，整体提升: {improvement:.2%}"
        else:
            reason = result['improvement']['overall']
            return False, f"新模型未提升，当前更好: {reason:.2%}"


# 单文件测试
if __name__ == '__main__':
    evaluator = ModelEvaluator()
    import sys
    if len(sys.argv) > 1:
        result = evaluator.evaluate_detection(sys.argv[1])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("用法: python model_evaluator.py <model_path>")