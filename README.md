# 斗地主AI助手

斗地主（Fight the Landlord）人工智能辅助系统，通过实时屏幕识别和AI决策算法，为玩家提供出牌建议。

## 功能特性

- **实时牌面识别**：基于YOLOv8的纯检测方案，无需模板匹配
- **多线程架构**：截图→识别→AI决策→UI，三线程并行处理
- **时序平滑**：5帧缓冲+60%多数投票，识别结果稳定不跳动
- **AI策略引擎**：支持激进/平衡/防御三种策略模式
- **DouZero集成**：可加载预训练DouZero ADP模型（需另行下载）

## 系统要求

- Windows 10/11
- Python 3.8+
- PyQt5
- PyTorch
- OpenCV
- YOLOv8 (ultralytics)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
python main.py
```

或编辑代码后使用模块重载模式：

```bash
python run_main.py
```

### AI测试

```bash
# 测试can_beat逻辑
python -c "import test_game; test_game.test_can_beat()"

# 运行完整模拟
python -c "import test_game; test_game.run_simulation()"

# 测试牌型识别
python test_detection.py
```

## 项目结构

```
.
├── main.py              # 主程序入口
├── run_main.py          # 模块重载入口
├── config.yaml          # 配置文件
├── window_capture.py    # 游戏窗口截取
├── card_recognizer.py   # 卡牌识别
├── state_manager.py     # 游戏状态管理
├── ai_engine.py         # AI决策引擎
├── ui.py                # 悬浮窗UI
├── modules/             # 核心模块
│   ├── window_capture.py
│   ├── card_recognizer.py
│   ├── state_manager.py
│   ├── ai_engine.py
│   └── ui.py
└── models/              # YOLO模型（需下载）
```

## 配置说明

`config.yaml` 中的主要配置项：

```yaml
game:
  type: "欢乐斗地主"           # 游戏类型
  window_title: ".*斗地主.*"   # 窗口标题（正则匹配）

recognition:
  fps: 1                     # 截图帧率
  confidence_threshold: 0.25 # 识别置信度阈值
  model_path: "models/yolov8_cards_v5.pt" # 当前模型

ai:
  strategy: "balanced"      # 策略：aggressive/balanced/defensive
  max_decision_time: 0.2     # 最大决策时间(秒)

ui:
  opacity: 0.9               # 窗口透明度
  position: "top-right"      # 窗口位置
  hotkey: "F1"               # 快捷键
```

## 模型下载

DouZero预训练模型需要单独下载（约6MB/个）：

```
douzero_models/
├── douzero_landlord.ckpt
├── douzero_landlord_up.ckpt
└── douzero_landlord_down.ckpt
```

下载地址：https://github.com/kwai/DouZero#pretrained-models

## AI策略测试结果

100局自我对战测试：

| 地主配置 | 农民配置 | 地主胜率 |
|----------|----------|----------|
| balanced | aggressive + defensive | 100% |
| aggressive | balanced + balanced | 60% |
| defensive | aggressive + aggressive | 22% |

**最优配置**：地主用`balanced`，农民用`aggressive + defensive`

## 卡牌编码

15类检测模型，牌面值对应：

| 类别 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|
| 牌面 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K | A | 2 | 小王 | 大王 |

AI引擎使用DouZero对齐的牌值编码：`3→3, 4→4, ..., 2→17, 小王→20, 大王→30`

## 技术栈

- **窗口捕获**：pywin32 (win32gui) + mss
- **卡牌检测**：YOLOv8 + CLAHE预处理
- **AI决策**：规则评分引擎 + DouZero ADP
- **UI界面**：PyQt5（无边框、置顶、半透明）
- **平台**：仅支持Windows

## License

MIT License
