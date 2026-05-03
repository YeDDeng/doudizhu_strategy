"""
迭代训练器 - 从当前最佳模型开始，持续改进

使用方法:
  python -c "from modules.iterative_trainer import IterativeTrainer; t = IterativeTrainer(); t.train()"
"""

import os
import sys
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class IterativeTrainer:
    """迭代训练器"""

    DEFAULT_CONFIG = {
        'epochs': 50,
        'batch': 4,
        'imgsz': 1280,
        'device': 'cpu',
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 20,
        'save_period': 10,
        'augment': True,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
    }

    def __init__(self, config: Dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # 路径配置
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.dataset_dir = self.project_root / "dataset"

        # 数据目录
        self.original_images = self.dataset_dir / "images" / "train"
        self.original_labels = self.dataset_dir / "labels" / "train"
        self.database_images = self.dataset_dir / "images" / "database"
        self.database_labels = self.dataset_dir / "labels" / "database"
        self.candidate_images = self.dataset_dir / "candidate" / "images" / "train"
        self.candidate_labels = self.dataset_dir / "candidate" / "labels" / "train"

        # 输出目录
        self.runs_dir = self.project_root / "runs" / "detect"

        # 注册表
        self.registry_path = self.models_dir / "MODEL_REGISTRY.json"

        # 加载注册表获取当前最佳
        registry = self._load_registry()
        self.current_best = registry.get('active_model', 'yolov8_cards.pt')
        self.current_best_path = self.models_dir / self.current_best

        # 训练历史
        self.training_history = []
        self.history_file = self.dataset_dir / "training_history.json"

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding='utf-8'))
        return {"models": [], "active_model": "yolov8_cards.pt"}

    def _save_registry(self, registry: Dict):
        self.registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding='utf-8')

    def _load_history(self):
        if self.history_file.exists():
            self.training_history = json.loads(self.history_file.read_text(encoding='utf-8'))

    def _save_history(self):
        self.history_file.write_text(json.dumps(self.training_history, indent=2, ensure_ascii=False), encoding='utf-8')

    def get_current_best(self) -> str:
        """获取当前最佳模型名称"""
        return self.current_best

    def get_candidates_count(self) -> int:
        """获取候选数据数量"""
        count = 0
        if self.candidate_images.exists():
            count += len(list(self.candidate_images.glob("*.png")))
            count += len(list(self.candidate_images.glob("*.jpg")))
        return count

    def get_database_count(self) -> int:
        """获取数据库图片数量"""
        count = 0
        if self.database_images.exists():
            count += len(list(self.database_images.glob("*.png")))
            count += len(list(self.database_images.glob("*.jpg")))
        return count

    def get_original_count(self) -> int:
        """获取原始训练数据数量"""
        count = 0
        if self.original_images.exists():
            count += len(list(self.original_images.glob("*.png")))
            count += len(list(self.original_images.glob("*.jpg")))
        return count

    def get_total_training_count(self) -> int:
        """获取总训练数据数量"""
        return self.get_original_count() + self.get_database_count() + self.get_candidates_count()

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'current_best': self.current_best,
            'current_best_path': str(self.current_best_path),
            'original_images': self.get_original_count(),
            'database_images': self.get_database_count(),
            'candidate_images': self.get_candidates_count(),
            'total_training': self.get_total_training_count(),
            'training_rounds': len(self.training_history),
            'config': self.config
        }

    def prepare_dataset_yaml(self, name: str = "combined") -> str:
        """准备合并数据集的yaml配置"""
        yaml_path = self.dataset_dir / f"dataset_{name}.yaml"

        # 动态构建train列表
        train_parts = [f"../images/train"]

        if self.database_images.exists() and any(self.database_images.glob("*")):
            train_parts.append("../images/database")

        if self.candidate_images.exists() and any(self.candidate_images.glob("*")):
            train_parts.append("../candidate/images/train")

        yaml_content = f"""# 迭代训练数据集 - {name}
# 自动生成

path: {str(self.dataset_dir.absolute())}
train:
  - images/train
  - images/database
  - candidate/images/train
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
"""

        yaml_path.write_text(yaml_content, encoding='utf-8')
        return str(yaml_path)

    def get_next_version(self) -> str:
        """获取下一个版本号"""
        registry = self._load_registry()
        existing_versions = []

        for model in registry.get('models', []):
            name = model.get('name', '')
            if name.startswith('yolov8_cards_v') and name.endswith('.pt'):
                try:
                    v = int(name.replace('yolov8_cards_v', '').replace('.pt', ''))
                    existing_versions.append(v)
                except:
                    pass

        next_v = max(existing_versions) + 1 if existing_versions else 5
        return f"yolov8_cards_v{next_v}.pt"

    def train(self,
              new_model_name: Optional[str] = None,
              epochs: Optional[int] = None,
              blocking: bool = True) -> Tuple[bool, str]:
        """
        执行迭代训练

        Args:
            new_model_name: 新模型名称，默认自动生成
            epochs: 训练轮数，默认使用配置值
            blocking: 是否阻塞等待完成

        Returns:
            (success, message)
        """
        if not ULTRALYTICS_AVAILABLE:
            return False, "ultralytics未安装"

        # 检查基础模型
        if not self.current_best_path.exists():
            return False, f"当前最佳模型不存在: {self.current_best_path}"

        # 统计数据量
        total = self.get_total_training_count()
        min_required = 50
        if total < min_required:
            return False, f"训练数据不足: {total} < {min_required}"

        # 确定模型名称
        if new_model_name is None:
            new_model_name = self.get_next_version()

        # 准备数据集
        dataset_yaml = self.prepare_dataset_yaml(new_model_name)

        # 训练参数
        train_epochs = epochs or self.config['epochs']

        print(f"[IterativeTrainer] 开始迭代训练")
        print(f"  当前最佳: {self.current_best}")
        print(f"  新模型: {new_model_name}")
        print(f"  训练数据: {total}张 (原始{self.get_original_count()} + 数据库{self.get_database_count()} + 候选{self.get_candidates_count()})")
        print(f"  训练轮数: {train_epochs}")
        print(f"  数据集: {dataset_yaml}")

        if not blocking:
            # 后台训练
            thread = threading.Thread(target=self._train_async,
                                       args=(new_model_name, dataset_yaml, train_epochs),
                                       daemon=True)
            thread.start()
            return True, f"训练已在后台开始: {new_model_name}"

        # 阻塞训练
        return self._train_async(new_model_name, dataset_yaml, train_epochs)

    def _train_async(self, new_model_name: str, dataset_yaml: str, train_epochs: int) -> Tuple[bool, str]:
        """异步执行训练"""
        try:
            # 加载模型
            model = YOLO(str(self.current_best_path))

            # 开始训练
            results = model.train(
                data=dataset_yaml,
                epochs=train_epochs,
                batch=self.config['batch'],
                imgsz=self.config['imgsz'],
                device=self.config['device'],
                lr0=self.config['lr0'],
                lrf=self.config['lrf'],
                patience=self.config['patience'],
                save_period=self.config['save_period'],
                augment=self.config['augment'],
                fliplr=self.config['fliplr'],
                mosaic=self.config['mosaic'],
                mixup=self.config['mixup'],
                project=str(self.runs_dir),
                name=f"iterative_{new_model_name.replace('.pt', '')}",
                exist_ok=True,
                verbose=True
            )

            # 获取训练好的模型
            save_dir = Path(results.save_dir)
            best_model_path = save_dir / "weights" / "best.pt"

            if not best_model_path.exists():
                best_model_path = save_dir / "weights" / "last.pt"

            if not best_model_path.exists():
                return False, "训练完成但未找到模型文件"

            # 复制到models目录
            final_model_path = self.models_dir / new_model_name
            shutil.copy2(str(best_model_path), str(final_model_path))

            # 更新注册表
            registry = self._load_registry()
            size_mb = final_model_path.stat().st_size / (1024 * 1024)

            entry = {
                "name": new_model_name,
                "version": new_model_name.replace("yolov8_cards_v", "").replace(".pt", ""),
                "added_at": datetime.now().strftime("%Y-%m-%d"),
                "source": str(best_model_path),
                "size_mb": round(size_mb, 1),
                "trained_on": f"{self.get_total_training_count()}张",
                "accuracy": "待评估",
                "notes": "迭代训练生成"
            }
            registry["models"].append(entry)
            self._save_registry(registry)

            # 记录历史
            self._load_history()
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'new_model': new_model_name,
                'base_model': self.current_best,
                'epochs': train_epochs,
                'total_data': self.get_total_training_count(),
                'size_mb': round(size_mb, 1),
                'status': 'success'
            })
            self._save_history()

            # 更新当前最佳
            old_best = self.current_best
            self.current_best = new_model_name
            self.current_best_path = self.models_dir / new_model_name

            # 清理临时yaml
            Path(dataset_yaml).unlink(missing_ok=True)

            print(f"[IterativeTrainer] 训练完成: {final_model_path}")
            print(f"  替换: {old_best} -> {new_model_name}")

            return True, str(final_model_path)

        except Exception as e:
            import traceback
            traceback.print_exc()

            self._load_history()
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'new_model': new_model_name,
                'base_model': self.current_best,
                'epochs': train_epochs,
                'status': 'failed',
                'error': str(e)
            })
            self._save_history()

            return False, f"训练失败: {e}"

    def get_history(self, limit: int = 20) -> list:
        """获取训练历史"""
        self._load_history()
        return self.training_history[-limit:]


# 单文件测试
if __name__ == '__main__':
    trainer = IterativeTrainer()
    print(json.dumps(trainer.get_status(), indent=2, ensure_ascii=False))