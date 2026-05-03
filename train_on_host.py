"""
斗地主AI助手 - 宿主机训练脚本
在Windows宿主机上运行，使用GPU训练

使用方法:
1. 确保 dataset/images/train 文件夹存在且包含63张标注数据
2. 确保 models/ 文件夹存在
3. 运行: python train_on_host.py
"""

import os
from datetime import datetime
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 创建训练配置文件
TRAIN_YAML = os.path.join(PROJECT_ROOT, 'dataset_host_train.yaml')
with open(TRAIN_YAML, 'w', encoding='utf-8') as f:
    f.write('''# 斗地主训练配置 - 宿主机训练用
path: {PROJECT_ROOT}/dataset
train: images/train
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
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_on_host')

def main():
    print("=" * 60)
    print("斗地主AI助手 - 宿主机训练")
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
        print("继续训练...")

    # 加载预训练模型 (使用small版本，精度更高)
    print("\n加载 YOLOv8s 预训练模型...")
    model = YOLO('yolov8s.pt')

    # 开始训练
    print(f"""
训练配置:
  - 数据集: dataset/images/train (63张图片，全部有标注)
  - 模型: yolov8s.pt (small, 比nano精度高)
  - 图像大小: 1280
  - 批次大小: 16 (GPU)
  - 训练轮次: 100 epochs
  - 启用数据增强:
    * 水平翻转: 0.5
    * 亮度: hsv_v=0.4
    * 缩放: scale=0.5
    * 旋转: degrees=10
  - 保存目录: {MODEL_SAVE_DIR}
""")

    results = model.train(
        data=DATA_CONFIG,
        epochs=100,
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
        # 数据增强 - 关键改进
        degrees=10.0,           # 旋转 ±10度
        translate=0.1,         # 平移 ±10%
        scale=0.5,             # 缩放 0.5-1.5
        hsv_h=0.015,          # 色调增强
        hsv_s=0.7,            # 饱和度增强
        hsv_v=0.4,            # 亮度增强
        fliplr=0.5,           # 水平翻转 (关键!)
        flipud=0.0,           # 垂直翻转(扑克牌不需要)
        mosaic=1.0,           # 马赛克增强
        mixup=0.1,            # MixUp增强
        copy_paste=0.0,       # Copy-paste增强
        # 训练策略
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 其他
        patience=50,          # 早停
        save=True,
        save_period=10,       # 每10轮保存
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
        backup_name = f"yolov8_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        backup_path = os.path.join(PROJECT_ROOT, 'models', backup_name)
        import shutil
        shutil.copy(best_model, backup_path)
        print(f"备份模型: {backup_path}")

        # 替换主模型
        main_model = os.path.join(PROJECT_ROOT, 'models', 'yolov8_cards.pt')
        shutil.copy(best_model, main_model)
        print(f"更新主模型: {main_model}")
    else:
        print("警告: 未找到最佳模型权重文件")

    print(f"\n训练结果保存在: {MODEL_SAVE_DIR}")
    print("打开查看 results.png 训练曲线")


if __name__ == '__main__':
    main()
