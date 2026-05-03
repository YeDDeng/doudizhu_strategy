"""
模型管理工具 - 训练新模型时保留旧模型

使用方法:
    python model_manager.py list          # 列出所有模型
    python model_manager.py use <name>    # 切换使用的模型
    python model_manager.py train <path>  # 添加新训练的模型
    python model_manager.py info <name>   # 查看模型信息
"""

import os
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path


MODELS_DIR = Path("models")
ACTIVE_FILE = MODELS_DIR / "active_model.txt"
MODEL_META_DIR = MODELS_DIR / ".metadata"


def ensure_dirs():
    """确保目录存在"""
    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_META_DIR.mkdir(exist_ok=True)


def get_active_model():
    """获取当前激活的模型"""
    if ACTIVE_FILE.exists():
        return ACTIVE_FILE.read_text().strip()
    # 默认返回yolov8_cards.pt
    return "yolov8_cards.pt"


def set_active_model(name):
    """设置激活的模型"""
    if not (MODELS_DIR / name).exists():
        print(f"错误: 模型 {name} 不存在")
        return False
    ACTIVE_FILE.write_text(name)
    print(f"已激活模型: {name}")
    return True


def list_models():
    """列出所有模型"""
    ensure_dirs()

    active = get_active_model()
    registry = _load_registry()

    print("\n" + "="*60)
    print("  斗地主AI - 模型列表")
    print("="*60)

    if not any(MODELS_DIR.glob("*.pt")):
        print("\n没有找到任何模型文件")
        return

    print(f"\n当前激活: {active}\n")

    # 按修改时间排序
    models = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    print(f"{'名称':<30} {'大小':>10} {'修改时间':>20}")
    print("-"*62)

    for m in models:
        if m.name.startswith('.'):
            continue
        size = m.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        else:
            size_str = f"{size/1024:.0f} KB"

        mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        active_mark = " ← 当前使用" if m.name == active else ""

        # 查找registry中的记录
        reg_entry = next((r for r in registry.get('models', []) if r['name'] == m.name), None)
        if reg_entry and 'accuracy' in reg_entry:
            acc = reg_entry['accuracy']
            active_mark += f" ({acc})"

        print(f"{m.name:<30} {size_str:>10} {mtime:>20}{active_mark}")

    print()

    # 显示历史摘要
    if registry.get('models'):
        print("模型历史:")
        for r in registry['models'][-3:]:
            print(f"  - {r.get('name')} ({r.get('added_at', 'unknown')}): {r.get('notes', '')}")
        print()


def _load_registry():
    """加载模型注册表"""
    registry_file = MODELS_DIR / "MODEL_REGISTRY.json"
    if registry_file.exists():
        try:
            return json.loads(registry_file.read_text(encoding='utf-8'))
        except:
            pass
    return {"models": [], "active_model": "yolov8_cards.pt"}


def _save_registry(registry):
    """保存模型注册表"""
    registry_file = MODELS_DIR / "MODEL_REGISTRY.json"
    registry_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding='utf-8')


def add_model(source_path, model_name=None, notes=""):
    """
    添加新模型 - 不删除旧模型

    Args:
        source_path: 源模型文件路径
        model_name: 可选，自定义名称。如果不提供则使用源文件名
        notes: 可选，关于这个模型的备注
    """
    ensure_dirs()

    source = Path(source_path)
    if not source.exists():
        print(f"错误: 源模型 {source_path} 不存在")
        return False

    # 确定目标名称
    if model_name is None:
        model_name = source.name

    dest = MODELS_DIR / model_name

    # 如果目标已存在，提示用户
    if dest.exists():
        print(f"错误: 模型 {model_name} 已存在")
        print("请使用不同的名称，或先切换活跃模型再添加")
        return False

    # 复制新模型
    shutil.copy2(str(source), str(dest))

    # 更新注册表
    registry = _load_registry()
    size_mb = dest.stat().st_size / (1024*1024)

    # 自动生成版本号
    existing = [r for r in registry.get('models', []) if 'yolov8_cards' in r.get('name', '')]
    version_num = len(existing) + 1

    entry = {
        "name": model_name,
        "version": f"v{version_num}",
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "source": str(source),
        "size_mb": round(size_mb, 1),
        "notes": notes or "新添加的模型"
    }
    registry["models"].append(entry)
    _save_registry(registry)

    print(f"已添加模型: {model_name}")
    print(f"  路径: {dest}")
    print(f"  大小: {size_mb:.2f} MB")
    print(f"  版本: v{version_num}")

    return True


def switch_model(name):
    """切换使用的模型"""
    ensure_dirs()

    target = MODELS_DIR / name
    if not target.exists():
        print(f"错误: 模型 {name} 不存在")
        print("\n可用模型:")
        list_models()
        return False

    set_active_model(name)

    # 更新config.yaml中的model_path
    config_path = Path("config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        old_path = config.get('recognition', {}).get('model_path', '')
        config['recognition']['model_path'] = f"models/{name}"

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print(f"已更新 config.yaml: model_path -> models/{name}")

    # 更新注册表
    registry = _load_registry()
    registry["active_model"] = name

    # 更新旧活跃模型的备注
    for m in registry.get("models", []):
        if m.get("name") == name:
            m["notes"] = m.get("notes", "") + " ← 当前活跃"
            break

    _save_registry(registry)

    print(f"已切换活跃模型: {name}")

    return True


def model_info(name):
    """显示模型详细信息"""
    target = MODELS_DIR / name
    if not target.exists():
        print(f"错误: 模型 {name} 不存在")
        return False

    print(f"\n{'='*50}")
    print(f"  模型信息: {name}")
    print(f"{'='*50}")
    print(f"  路径: {target}")
    print(f"  大小: {target.stat().st_size / (1024*1024):.2f} MB")
    print(f"  修改: {datetime.fromtimestamp(target.stat().st_mtime)}")

    # 尝试获取模型信息
    try:
        from ultralytics import YOLO
        model = YOLO(str(target))
        print(f"  层数: {len(model.model.model)}")
        print(f"  参数: {sum(p.numel() for p in model.model.parameters()):,}")
    except Exception as e:
        print(f"  (无法加载模型详情: {e})")

    # 元数据
    meta_file = MODEL_META_DIR / f"{name}.json"
    if meta_file.exists():
        print("\n  元数据:")
        meta = json.loads(meta_file.read_text())
        for k, v in meta.items():
            if k != 'size':
                print(f"    {k}: {v}")

    print()
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        list_models()
        return

    cmd = sys.argv[1].lower()

    if cmd == "list":
        list_models()

    elif cmd == "use":
        if len(sys.argv) < 3:
            print("用法: python model_manager.py use <模型名称>")
            return
        switch_model(sys.argv[2])

    elif cmd == "add":
        if len(sys.argv) < 3:
            print("用法: python model_manager.py add <模型路径> [名称]")
            return
        name = sys.argv[3] if len(sys.argv) > 3 else None
        add_model(sys.argv[2], name)

    elif cmd == "train":
        # 训练后自动添加模型
        if len(sys.argv) < 3:
            print("用法: python model_manager.py train <新模型路径>")
            return
        new_model = sys.argv[2]
        add_model(new_model)

    elif cmd == "info":
        if len(sys.argv) < 3:
            print("用法: python model_manager.py info <模型名称>")
            return
        model_info(sys.argv[2])

    elif cmd == "switch":
        if len(sys.argv) < 3:
            print("用法: python model_manager.py switch <模型名称>")
            return
        switch_model(sys.argv[2])

    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
