"""
训练管理器 - 控制迭代训练循环流程

整合数据标注和模型训练
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable

from .iterative_trainer import IterativeTrainer
from .model_evaluator import ModelEvaluator


class TrainingManager:
    """训练管理器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

        self.trainer = IterativeTrainer()
        self.evaluator = ModelEvaluator()

        self._is_training = False
        self._is_labeling = False
        self._training_thread = None

        # 回调
        self._on_training_complete = None
        self._on_labeling_progress = None

    def get_status(self) -> Dict:
        """获取当前状态"""
        status = self.trainer.get_status()

        # 检查注册表
        registry_path = self.project_root / "models" / "MODEL_REGISTRY.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding='utf-8'))
            status['models_count'] = len(registry.get('models', []))
            status['active_model'] = registry.get('active_model', 'unknown')

        status['is_training'] = self._is_training
        status['is_labeling'] = self._is_labeling

        return status

    def get_history(self, limit: int = 20) -> list:
        """获取训练历史"""
        return self.trainer.get_history(limit)

    def set_on_training_complete(self, callback: Callable):
        self._on_training_complete = callback

    def set_on_labeling_progress(self, callback: Callable):
        self._on_labeling_progress = callback

    def run_batch_labeling(self, limit: int = 0, blocking: bool = True) -> Dict:
        """
        运行批量标注

        Args:
            limit: 处理数量限制，0=全部
            blocking: 是否阻塞

        Returns:
            {'success': bool, 'message': str}
        """
        try:
            import sys
            sys.path.insert(0, str(self.project_root))
            from batch_labeler import main as labeler_main

            print("[TrainingManager] 开始批量标注...")

            if blocking:
                # 设置进度回调
                import argparse
                args = argparse.Namespace(
                    dir=str(Path("C:/Users/30330/Desktop/斗地主数据库")),
                    limit=limit,
                    model='models/yolov8_cards.pt',
                    skip_labeled=True
                )

                # 直接运行标注
                self._is_labeling = True
                try:
                    # 导入并准备参数
                    import os
                    DB_DIR = Path("C:/Users/30330/Desktop/斗地主数据库")
                    OUTPUT_LABEL_DIR = self.project_root / "dataset" / "labels" / "database"
                    OUTPUT_IMAGE_DIR = self.project_root / "dataset" / "images" / "database"

                    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
                    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

                    files = sorted(os.listdir(DB_DIR))
                    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    # 过滤已标注
                    unlabeled = []
                    for fname in image_files:
                        label_path = OUTPUT_LABEL_DIR / (Path(fname).stem + '.txt')
                        if not label_path.exists():
                            unlabeled.append(fname)

                    if limit > 0:
                        unlabeled = unlabeled[:limit]

                    print(f"待标注: {len(unlabeled)} 张")

                    # 动态导入batch_labeler的函数
                    from batch_labeler import label_image

                    success = 0
                    failed = 0
                    for i, fname in enumerate(unlabeled):
                        img_path = DB_DIR / fname
                        print(f"[{i+1}/{len(unlabeled)}] {fname}")

                        try:
                            if label_image(str(img_path), 'models/yolov8_cards.pt',
                                          str(OUTPUT_LABEL_DIR), str(OUTPUT_IMAGE_DIR)):
                                success += 1
                            else:
                                failed += 1
                        except Exception as e:
                            print(f"  [ERROR] {e}")
                            failed += 1

                        import time
                        time.sleep(0.5)

                        if self._on_labeling_progress:
                            self._on_labeling_progress(i+1, len(unlabeled))

                    print(f"\n标注完成: 成功{success}, 失败{failed}")
                    return {'success': True, 'labeled': success, 'failed': failed}

                finally:
                    self._is_labeling = False

            else:
                # 后台运行
                def run_labeling():
                    self._is_labeling = True
                    # ... 标注逻辑
                    self._is_labeling = False

                thread = threading.Thread(target=run_labeling, daemon=True)
                thread.start()
                return {'success': True, 'message': '标注已在后台开始'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def start_training(self,
                       new_model_name: Optional[str] = None,
                       epochs: Optional[int] = None,
                       blocking: bool = False) -> Dict:
        """
        开始训练

        Args:
            new_model_name: 新模型名称
            epochs: 训练轮数
            blocking: 是否阻塞

        Returns:
            {'success': bool, 'message': str}
        """
        if self._is_training:
            return {'success': False, 'message': '训练已在进行中'}

        self._is_training = True

        def training_task():
            try:
                success, result = self.trainer.train(
                    new_model_name=new_model_name,
                    epochs=epochs,
                    blocking=True
                )

                if success:
                    # 评估新模型
                    current_best = self.trainer.get_current_best()
                    new_model = result

                    # 替换旧模型路径
                    old_model_path = self.project_root / "models" / current_best
                    new_model_path = Path(result)

                    should_update, reason = self.evaluator.should_update_best(
                        str(new_model_path),
                        str(old_model_path)
                    )

                    if should_update:
                        print(f"[TrainingManager] {reason}")
                    else:
                        print(f"[TrainingManager] 未采用新模型: {reason}")

                    if self._on_training_complete:
                        self._on_training_complete({
                            'success': success,
                            'new_model': str(new_model_path),
                            'should_update': should_update,
                            'reason': reason
                        })
                else:
                    print(f"[TrainingManager] 训练失败: {result}")

            finally:
                self._is_training = False

        if blocking:
            training_task()
            return {'success': True, 'message': '训练完成'}
        else:
            self._training_thread = threading.Thread(target=training_task, daemon=True)
            self._training_thread.start()
            return {'success': True, 'message': '训练已在后台开始'}

    def update_active_model(self, new_model_name: str) -> bool:
        """手动更新活跃模型"""
        try:
            registry_path = self.project_root / "models" / "MODEL_REGISTRY.json"
            registry = json.loads(registry_path.read_text(encoding='utf-8'))
            registry['active_model'] = new_model_name
            registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding='utf-8')

            # 更新trainer的当前最佳
            self.trainer.current_best = new_model_name
            self.trainer.current_best_path = self.project_root / "models" / new_model_name

            return True
        except Exception as e:
            print(f"更新活跃模型失败: {e}")
            return False