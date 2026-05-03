"""
斗地主AI助手 - 从 v1 继续训练
在Windows宿主机上运行，使用GPU训练

使用方法:
1. 确保 models/yolov8_cards_v1.pt 存在
2. 确保 dataset 目录包含所有标注数据 (原始88 + 数据库111)
3. 运行: python train_from_v1.py
"""

import os
from datetime import datetime
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 创建训练配置文件 - 合并所有数据
TRAIN_YAML = os.path.join(PROJECT_ROOT, 'dataset_merged.yaml')
with open(TRAIN_YAML, 'w', encoding='utf-8') as f:
    f.write('''# 斗地主训练配置 - 合并数据集
# 使用: 原始88张 + 数据库111张 + 候选数据
path: {PROJECT_ROOT}/dataset
train:
  - images/train
  - images/database
  - candidate/images/train
val: images/train

# 15个rank类别
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
'''.format(PROJECT_ROOT=PROJECT_ROOT.replace('\\\\', '/')))

DATA_CONFIG = TRAIN_YAML
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_from_v1')

def main():
    print("=" * 60)
    print("斗地主AI助手 - 从 v1 继续训练")
    print("=" * 60)

    # 检查GPU
    import torch
    if torch.cuda.is_available():
        device = '0'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        device = 'cpu'
        print("警告: 未检测到GPU，使用CPU训练（非常慢）")
        return

    # 检查 v1 模型
    v1_model = os.path.join(PROJECT_ROOT, 'models', 'yolov8_cards_v1.pt')
    if not os.path.exists(v1_model):
        print(f"错误: 找不到 v1 模型: {v1_model}")
        return
    print(f"基础模型: {v1_model}")

    # 加载 v1 模型 (作为预训练)
    print("\n加载 v1 预训练模型...")
    model = YOLO(v1_model)

    # 开始训练
    print(f"""
训练配置:
  - 基础模型: yolov8_cards_v1.pt (召回率100%)
  - 数据集: dataset/images/train + database + candidate (共199+张)
  - 图像大小: 1280
  - 批次大小: 16 (GPU)
  - 训练轮次: 80 epochs (早停patience=30)
  - 启用数据增强:
    * 水平翻转: 0.5
    * 亮度: hsv_v=0.4
    * 缩放: scale=0.5
    * 旋转: degrees=10
    * 马赛克: mosaic=1.0
    * MixUp: mixup=0.1
  - 保存目录: {MODEL_SAVE_DIR}
""")

    results = model.train(
        data=DATA_CONFIG,
        epochs=80,
        imgsz=1280,
        batch=16,
        device=device,
        project=os.path.dirname(MODEL_SAVE_DIR),
        name=os.path.basename(MODEL_SAVE_DIR),
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
        # 其他
        patience=30,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 复制最佳模型到 models 文件夹
    best_model = os.path.join(MODEL_SAVE_DIR, 'weights', 'best.pt')
    if os.path.exists(best_model):
        backup_name = f"yolov8_cards_v5.pt"
        backup_path = os.path.join(PROJECT_ROOT, 'models', backup_name)
        import shutil
        shutil.copy(best_model, backup_path)
        print(f"新模型已保存: {backup_path}")

        # 可选: 设为活跃模型
        # main_model = os.path.join(PROJECT_ROOT, 'models', 'yolov8_cards.pt')
        # shutil.copy(best_model, main_model)
        # print(f"已更新活跃模型: {main_model}")
    else:
        print("警告: 未找到最佳模型权重文件")

    print(f"\n训练结果保存在: {MODEL_SAVE_DIR}")
    print("打开查看 results.png 训练曲线")


if __name__ == '__main__':
    main()