"""
CPU训练脚本 - 从v1继续训练
参数针对CPU优化
"""

import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent

# 创建训练配置
TRAIN_YAML = PROJECT_ROOT / 'dataset_cpu_train.yaml'
TRAIN_YAML.write_text('''# 斗地主训练配置 - CPU训练
path: {PROJECT_ROOT}/dataset
train:
  - images/train
  - images/database
val: images/train

nc: 15
names:
  0: 3
  1: 4
  2: 5
  3: 6
  4: 7
  5: 8
  6: 9
  7: 10
  8: J
  9: Q
  10: K
  11: A
  12: 2
  13: Joker_B
  14: Joker_R
'''.format(PROJECT_ROOT=str(PROJECT_ROOT).replace('\\\\', '/')))

MODEL_SAVE_DIR = PROJECT_ROOT / 'runs' / 'detect' / 'train_cpu_v1'

def main():
    print("=" * 60)
    print("斗地主 - CPU训练 (从v1)")
    print("=" * 60)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"设备: CPU")

    v1_model = PROJECT_ROOT / 'models' / 'yolov8_cards_v1.pt'
    if not v1_model.exists():
        print(f"错误: 找不到 {v1_model}")
        return

    print(f"基础模型: {v1_model}")
    print(f"训练配置: CPU优化")
    print(f"  - batch: 4 (CPU内存限制)")
    print(f"  - imgsz: 640 (降低分辨率加快速度)")
    print(f"  - epochs: 30")
    print(f"  - patience: 10")

    model = YOLO(str(v1_model))

    results = model.train(
        data=str(TRAIN_YAML),
        epochs=30,
        imgsz=640,
        batch=4,
        device='cpu',
        project=str(MODEL_SAVE_DIR.parent),
        name=MODEL_SAVE_DIR.name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        # 数据增强
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # 训练策略
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=10,
        save=True,
        save_period=5,
        plots=True,
        val=True,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    best_model = MODEL_SAVE_DIR / 'weights' / 'best.pt'
    if not best_model.exists():
        best_model = MODEL_SAVE_DIR / 'weights' / 'last.pt'

    if best_model.exists():
        new_model = PROJECT_ROOT / 'models' / 'yolov8_cards_v5.pt'
        import shutil
        shutil.copy(str(best_model), str(new_model))
        print(f"新模型已保存: {new_model}")

    # 清理临时yaml
    TRAIN_YAML.unlink(missing_ok=True)

    print(f"训练结果: {MODEL_SAVE_DIR}")

if __name__ == '__main__':
    main()