---
name: model-version-manager
description: 管理YOLO模型版本 - 训练新模型时保留旧版本，不删除只添加
triggers:
  - /模型管理
  - /model
  - 训练新模型
  - 添加新模型
---

# 模型版本管理技能

## 核心规则

**绝对禁止删除任何旧模型文件。只能添加新模型，保留历史版本。**

原因：模型是训练成果的结晶，即使效果不佳也可能有保留价值。删除后无法恢复。

## 模型目录结构

```
models/
├── yolov8_cards.pt          # 当前活跃模型
├── yolov8_cards_v1.pt      # 历史版本
├── yolov8_cards_v2.pt      # 历史版本
└── ...
```

## 可用命令

| 命令 | 说明 |
|------|------|
| `/model list` | 列出所有模型及状态 |
| `/model add <path>` | 添加新模型到仓库 |
| `/model use <name>` | 切换活跃模型 |
| `/model info <name>` | 查看模型详情 |
| `/model history` | 查看模型历史记录 |

## 工作流程

### 1. 训练新模型后

训练完成后，**绝不**直接覆盖 `yolov8_cards.pt`。按照以下步骤：

```bash
# 1. 给新模型起一个版本名
# 2. 将新模型添加到models目录（保留旧模型）
cp runs/evolve/cards_v2/weights/best.pt models/yolov8_cards_v3.pt

# 3. 可选：测试新模型效果
python test_detection.py --model models/yolov8_cards_v3.pt

# 4. 如果新模型效果好，再切换为活跃模型
/model use yolov8_cards_v3.pt
```

### 2. 版本命名规范

| 类型 | 格式 | 示例 |
|------|------|------|
| 主版本 | `yolov8_cards_v{n}.pt` | `yolov8_cards_v3.pt` |
| 实验版 | `yolov8_cards_exp{n}.pt` | `yolov8_cards_exp1.pt` |
| 数据集 | `yolov8_cards_{数据集}.pt` | `yolov8_cards_doudizhu.pt` |

### 3. 模型元数据记录

每次添加新模型时，更新 `models/MODEL_REGISTRY.json`：

```json
{
  "models": [
    {
      "name": "yolov8_cards.pt",
      "version": "main",
      "added_at": "2026-04-25",
      "trained_on": "63张标注",
      "accuracy": "17/17 检测正确",
      "notes": "当前活跃模型"
    },
    {
      "name": "yolov8_cards_v1.pt", 
      "version": "v1",
      "added_at": "2026-04-20",
      "trained_on": "15张标注",
      "accuracy": "15/15 检测正确",
      "notes": "首次训练版本"
    }
  ]
}
```

## 注意事项

1. **不要覆盖**：`yolov8_cards.pt` 是入口点，但旧版本都应保留
2. **不要删除**：即使某个版本效果不好，也保留以便对比
3. **记录变化**：每次更新都记录改进点和问题
4. **备份重要版本**：对于特别重要的突破版本，可以额外备份到其他位置

## 自动执行清单

训练新模型后，系统自动：

1. ✓ 检查 `models/` 目录，确认旧模型未被删除
2. ✓ 新模型以版本号命名保存
3. ✓ 更新 `MODEL_REGISTRY.json`
4. ✓ 对比新旧模型效果
5. ✓ 用户确认后才切换活跃模型
