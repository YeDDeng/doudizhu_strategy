#!/usr/bin/env python3
"""
迭代训练主入口 - CLI命令行工具

使用方法:
  python iterative_train.py status                    # 查看状态
  python iterative_train.py label                      # 批量标注数据库
  python iterative_train.py label --limit 10          # 标注前10张（测试）
  python iterative_train.py train                      # 开始训练
  python iterative_train.py train --epochs 30         # 指定epoch
  python iterative_train.py history                    # 查看训练历史
  python iterative_train.py compare                     # 对比当前最佳vs新模型
"""

import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.training_manager import TrainingManager
from modules.iterative_trainer import IterativeTrainer
from modules.model_evaluator import ModelEvaluator


def cmd_status(manager: TrainingManager):
    """查看状态"""
    status = manager.get_status()

    print("\n" + "=" * 60)
    print("迭代训练系统状态")
    print("=" * 60)
    print(f"当前最佳模型: {status.get('current_best', 'unknown')}")
    print(f"活跃模型: {status.get('active_model', 'unknown')}")
    print(f"已注册模型数: {status.get('models_count', 0)}")
    print()
    print("训练数据:")
    print(f"  原始数据: {status.get('original_images', 0)} 张")
    print(f"  数据库图片: {status.get('database_images', 0)} 张")
    print(f"  候选数据: {status.get('candidate_images', 0)} 张")
    print(f"  总计: {status.get('total_training', 0)} 张")
    print()
    print("训练状态:")
    print(f"  正在训练: {'是' if status.get('is_training') else '否'}")
    print(f"  正在标注: {'是' if status.get('is_labeling') else '否'}")
    print(f"  已完成训练轮次: {status.get('training_rounds', 0)}")
    print()
    print("训练配置:")
    config = status.get('config', {})
    print(f"  epochs: {config.get('epochs', 0)}")
    print(f"  imgsz: {config.get('imgsz', 0)}")
    print(f"  device: {config.get('device', 'unknown')}")
    print("=" * 60)


def cmd_label(manager: TrainingManager, args):
    """批量标注数据库"""
    print("\n" + "=" * 60)
    print("批量标注斗地主数据库")
    print("=" * 60)

    limit = args.limit if hasattr(args, 'limit') else 0

    result = manager.run_batch_labeling(limit=limit, blocking=True)

    if result.get('success'):
        print(f"\n标注完成: 成功{result.get('labeled', 0)}张, 失败{result.get('failed', 0)}张")
    else:
        print(f"\n标注失败: {result.get('error', 'unknown')}")


def cmd_train(manager: TrainingManager, args):
    """开始训练"""
    print("\n" + "=" * 60)
    print("开始迭代训练")
    print("=" * 60)

    # 检查数据量
    status = manager.get_status()
    total = status.get('total_training', 0)
    min_required = 50

    if total < min_required:
        print(f"错误: 训练数据不足 ({total} < {min_required})")
        print("请先运行 'python iterative_train.py label' 标注数据库")
        return

    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else None
    new_model_name = args.name if hasattr(args, 'name') and args.name else None

    result = manager.start_training(
        new_model_name=new_model_name,
        epochs=epochs,
        blocking=True
    )

    print(f"\n训练结果: {result.get('message')}")


def cmd_history(manager: TrainingManager):
    """查看训练历史"""
    history = manager.get_history()

    print("\n" + "=" * 60)
    print("训练历史")
    print("=" * 60)

    if not history:
        print("暂无训练记录")
    else:
        for i, h in enumerate(reversed(history)):
            print(f"\n[{len(history) - i}] {h.get('timestamp', 'unknown')}")
            print(f"  新模型: {h.get('new_model', 'unknown')}")
            print(f"  基础模型: {h.get('base_model', 'unknown')}")
            print(f"  epochs: {h.get('epochs', 0)}")
            print(f"  训练数据: {h.get('total_data', 0)} 张")
            print(f"  状态: {h.get('status', 'unknown')}")
            if 'error' in h:
                print(f"  错误: {h.get('error')}")

    print("=" * 60)


def cmd_compare(manager: TrainingManager):
    """对比模型"""
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)

    trainer = IterativeTrainer()
    evaluator = ModelEvaluator()

    current_best = trainer.get_current_best()
    current_best_path = trainer.models_dir / current_best

    # 获取最新的训练模型
    registry_path = trainer.registry_path
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        models = registry.get('models', [])
        if models:
            latest_model = models[-1]['name']
            latest_path = trainer.models_dir / latest_model

            if latest_path != current_best_path and latest_path.exists():
                print(f"对比: {latest_model} vs {current_best}")
                result = evaluator.compare_models(str(latest_path), str(current_best_path))

                print(f"\n整体平均置信度:")
                print(f"  新模型: {result['new_model'].get('avg_confidence', 0):.4f}")
                print(f"  当前: {result['old_model'].get('avg_confidence', 0):.4f}")
                print(f"  提升: {result['improvement']['overall']:.4f}")

                print(f"\n每类别置信度:")
                print(f"{'类别':<10} {'新模型':<10} {'当前':<10} {'差异':<10}")
                print("-" * 40)
                for cname, stats in result['improvement']['per_class'].items():
                    diff = stats['diff']
                    marker = "+" if diff > 0 else "" if diff == 0 else ""
                    print(f"{cname:<10} {stats['new']:<10.4f} {stats['old']:<10.4f} {marker}{diff:.4f}")

                print(f"\n是否更好: {'是' if result['is_better'] else '否'}")
            else:
                print("没有找到需要对比的新模型")
        else:
            print("注册表中没有模型记录")
    else:
        print("注册表不存在")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='斗地主AI - 迭代训练系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python iterative_train.py status                    # 查看状态
  python iterative_train.py label                    # 标注数据库所有图片
  python iterative_train.py label --limit 10          # 只标注前10张(测试)
  python iterative_train.py train                      # 开始训练
  python iterative_train.py train --epochs 30         # 指定30轮
  python iterative_train.py train --name v6.pt        # 指定模型名
  python iterative_train.py history                   # 查看历史
  python iterative_train.py compare                   # 对比模型
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # status命令
    subparsers.add_parser('status', help='查看当前状态')

    # label命令
    label_parser = subparsers.add_parser('label', help='批量标注数据库图片')
    label_parser.add_argument('--limit', '-l', type=int, default=0, help='处理数量限制(0=全部)')

    # train命令
    train_parser = subparsers.add_parser('train', help='开始训练')
    train_parser.add_argument('--epochs', '-e', type=int, help='训练轮数')
    train_parser.add_argument('--name', '-n', type=str, help='新模型名称')

    # history命令
    subparsers.add_parser('history', help='查看训练历史')

    # compare命令
    subparsers.add_parser('compare', help='对比模型')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化管理器
    manager = TrainingManager()

    if args.command == 'status':
        cmd_status(manager)
    elif args.command == 'label':
        cmd_label(manager, args)
    elif args.command == 'train':
        cmd_train(manager, args)
    elif args.command == 'history':
        cmd_history(manager)
    elif args.command == 'compare':
        cmd_compare(manager)


if __name__ == '__main__':
    main()