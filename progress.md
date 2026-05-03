# 斗地主AI助手 - 开发进度记录

## 项目概述
使用YOLOv8进行实时卡牌检测、基于规则的AI决策引擎、PyQt5悬浮窗显示的斗地主辅助工具。

---

## 已完成的重大修复

### 1. 悬浮窗拖动后按钮失灵 [已解决]
- **文件**: `modules/ui.py`
- **方案**: 使用 `nativeEvent` 拦截 `WM_NCHITTEST` 消息，返回 `HTCAPTION(2)` 让Windows原生处理拖动
- **效果**: 拖动完全不干扰Qt事件循环，按钮始终可点击

### 2. AI总是建议王炸 [已解决]
- **文件**: `modules/ai_engine.py`
- **方案**: `rocket` 降至20，`bomb` 降至15
- **效果**: AI不再滥用王炸

### 3. AI用炸弹管单牌/对子 [已解决]
- **文件**: `modules/ai_engine.py: _evaluate_play()`
- **修复**: 添加"最小胜出原则" - 对普通牌型使用炸弹-30分、王炸-20分的惩罚
- **效果**: AI优先使用同类型牌管牌

### 4. 检测数字跳动(18→4→9) [已解决]
- **文件**: `modules/card_recognizer.py`
- **修复**: 改用"卡牌数量共识" - 统计历史帧的卡牌数量，取出现次数最多的(≥60%阈值)

### 5. 去重逻辑优化 [2026-04-25 新完成]
- **文件**: `modules/card_recognizer.py`
- **修复**:
  - 启用 `_cluster_detections()` - IoU-based NMS合并重叠框（阈值0.35）
  - 新增 `_deduplicate_by_position()` - 40px范围内按位置去重
  - CLAHE预处理仅对暗色图片(亮度<80)应用
  - imgsz从640改为1280提高检测质量

### 6. Joker检测增强 [2026-04-25 新完成]
- **文件**: `modules/card_recognizer.py`
- **修复**:
  - `_supplement_jokers()` 用conf=0.03二次检测
  - `_verify_and_correct_jokers_by_color()` 颜色验证纠正
  - `_supplement_low_confidence_cards()` - 检测<15张时用conf=0.015补检K、2、8

---

## 模型状态

- 当前模型: `models/yolov8_cards.pt` (24MB)
- 来源: `runs/detect/train_cards_v6/weights/best.pt`
- 迁移学习尝试失败: 63张训练数据不足以提升，反而退化

---

## 已修复Bug

### config.yaml策略配置无效 [2026-04-25 修复]
- **文件**: `main.py:52`
- **问题**: 读取 `config.get('strategy_mode')` 但配置是 `ai.strategy`，导致AI策略永远是balanced
- **修复**: 改为 `config.get('ai', {}).get('strategy', 'balanced')`

---

## 训练方案 (P1)

### 数据集分析
| 文件夹 | 图片 | 标注 | 配对 |
|--------|------|------|------|
| images/train | 63 | 88 | **63** (全部配对) |
| images/new | 33 | 89 | 33 |

### 训练脚本
- **文件**: `train_on_host.py` (新建)
- **数据集**: images/train (63张，全部有标注)
- **模型**: yolov8s.pt (small，比nano精度高)
- **增强**: fliplr=0.5, degrees=10, scale=0.5, hsv_v=0.4, mixup=0.1
- **GPU batch**: 16
- **预计时间**: 1-2小时 (GPU)

### 使用方法
1. 宿主机安装: `pip install ultralytics torch torchvision`
2. 确认 dataset 文件夹存在
3. 运行: `python train_on_host.py`

---

## 测试结果

| 图片 | 实际 | 检测 | 漏检 | 误检 |
|------|------|------|------|------|
| 微信图片_20260415202911_90_16.jpg | 17张 | 17张 | 0 | 0 |
| 微信图片_20260410223605_8_5.png | 18张 | 16张 | K, 2, 8 | 0 |

### 图片2检测问题
- K(0.05)、2(0.01)、8(0.02) 置信度极低
- 原因: 模型固有限制，低置信度下捕获会引入大量误检

---

## config.yaml 当前配置

```yaml
recognition:
  fps: 8
  confidence_threshold: 0.08  # 建议值
  model_path: "models/yolov8_cards.pt"
```

---

## 待优化项

1. **Q/10混淆** - `_filter_q10_confusion()` 逻辑可进一步优化
2. **时序平滑** - 当前历史窗口5帧，可考虑增加
3. **2/8/K检测** - 需更多训练样本提升模型能力
4. **误检控制** - conf=0.05时误检率上升，建议保持0.08-0.15

---

## 关键文件清单

| 文件 | 职责 | 最近修改 |
|------|------|---------|
| `main.py` | 主程序入口，3线程协调 | 添加 `started` 标志控制按钮 |
| `modules/ui.py` | PyQt5悬浮窗 | nativeEvent拖动方案 |
| `modules/card_recognizer.py` | YOLO检测+时间平滑 | 去重逻辑、CLAHE优化、imgsz1280 |
| `modules/ai_engine.py` | AI决策引擎 | 最小胜出原则、分数调整 |
| `modules/state_manager.py` | 游戏状态管理 | Pass检测 |
| `modules/window_capture.py` | 窗口捕获 | 无重大修改 |
| `test_game.py` | 测试系统 | 多轮测试用例 |
| `config.yaml` | 配置文件 | fps=8, confidence=0.08 |

---

## 下一步建议

1. **[进行中] 模型重训练**: 使用 train_on_host.py 在宿主机GPU上训练
2. **实际游戏测试**: 训练完成后在真实游戏环境中验证2/8/K检测改善
3. **增强后处理**: 对检测出的牌型进行合法性校验
4. **复杂牌型测试**: 测试顺子、飞机、三带一等复杂牌型的AI决策
