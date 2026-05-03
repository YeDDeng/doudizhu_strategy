# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Doudizhu (Fight the Landlord) AI Assistant** - a real-time computer vision and AI system that provides strategy suggestions for the Chinese card game "Fight the Landlord". The system captures the game window, recognizes cards using **YOLOv8 pure detection** (no template matching), tracks game state via temporal smoothing, and displays AI recommendations through a floating PyQt5 window.

## Architecture

```
Game Window
    ↓
WindowCapture (win32gui + mss, fps=1, rect移动平均平滑)
    ↓ frame_queue(maxsize=2)
CardRecognizer (sequential单模型推理, imgsz=480)
  ├─ 手牌区域独立YOLO推理 (480×480)
  ├─ 对手区域独立YOLO推理 (conf=0.01, 低阈值)
  ├─ 后处理 (聚类/去重/颜色校验/补充检测)
  ├─ 时序平滑 (5帧缓冲, 60%多数投票)
  └─ 识别结果稳定性检查 (数量+组成变化才更新)
    ↓
GameStateManager (54张牌追踪, 出牌历史, 牌型识别)
    ↓
HybridAI ─┬─ DouZeroRealTime (~3-8ms, ADP模型)
          └─ DoudizhuAI (~60ms, 规则AI评分引擎)
    ↓ result_queue(maxsize=1)
AIFloatingWindow (PyQt5, 100ms QTimer轮询, 决策变化才刷新建议)
```

**3-线程数据流：**
- **截图线程** → `frame_queue` (maxsize=2) → **处理线程** → `result_queue` (maxsize=1) → **UI线程** (QTimer 每100ms轮询)
- 截图循环等待 `self.started == True`（用户点击"开始识别"后才启动）
- 处理线程从 frame_queue 取帧，执行识别+AI决策，结果放入 result_queue
- UI 线程通过 QTimer 轮询 result_queue 更新悬浮窗

**启动入口：**
- `main.py` — 标准入口，优先导入 torch 避免 DLL 问题
- `run_main.py` — 模块重载入口，编辑后运行此文件可刷新代码

### Modules

| Module | File | Responsibility |
|--------|------|----------------|
| WindowCapture | `window_capture.py` | Find game window via win32gui, capture screenshots with mss |
| CardRecognizer | `card_recognizer.py` | Per-region cropping + parallel YOLOv8 + temporal smoothing (5-frame) |
| GameStateManager | `state_manager.py` | Track 54-card deck, play history, card types |
| HybridAI + DoudizhuAI | `ai_engine.py` | DouZero (3-8ms) fallback to rule-based scoring engine (60ms) |
| AIFloatingWindow | `ui.py` | PyQt5 always-on-top draggable translucent window |
| DoudizhuAssistant | `main.py` | Orchestrate 3 threads, queues, config loading |
| IterativeTrainer | `modules/iterative_trainer.py` | Iterative training from current best model, dataset yaml preparation |
| TrainingManager | `modules/training_manager.py` | High-level controller for labeling + training loop |
| ModelEvaluator | `modules/model_evaluator.py` | Compare model accuracy, decide whether to adopt new models |

### Additional Scripts

| Script | Purpose |
|--------|---------|
| `multi_agent_game.py` | Multi-agent self-play learning system with genetic algorithm optimization |
| `game_referee.py` | Game referee simulator for AI vs AI matches |
| `run_100_games.py` | Batch testing script for 100-game statistics |
| `douzero_ai.py` | DouZero integration framework (requires pretrained models) |
| `strategy_tournament.py` | Simplified strategy comparison (aggressive/balanced/defensive) |

### DouZero Models (Separate Download Required)

DouZero ADP models (`~6MB each`) are **not included in the repo**. Place them at:
- `douzero_models/douzero_landlord.ckpt`
- `douzero_models/douzero_landlord_up.ckpt`
- `douzero_models/douzero_landlord_down.ckpt`

Without these files, `HybridAI` auto-falls back to the rule-based `DoudizhuAI` engine.

### Class-to-Rank Mapping (15 classes, used in YOLO and all dataset YAML files)

```
0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10,
8: J, 9: Q, 10: K, 11: A, 12: 2, 13: Joker_B, 14: Joker_R
```

### DouZero-Aligned Card Value Encoding

AI engine converts card strings to integer values using `_card_to_env(card: str) → int`:
- Ranks map to: `3→3, 4→4, ..., 10→10, J→11, Q→12, K→13, A→14, 2→17, Joker_B→20, Joker_R→30`
- Suit info is dropped (only rank matters for comparison)
- Joker variants (`Joker_B`, `Joker_R`, `JB`, `JR`) all normalize to the same env values
- The `CARD_VALUE` dict in both `ai_engine.py` and `state_manager.py` maintains this mapping, aligning with DouZero environment values for compatibility.

### HybridAI Engine

`create_ai_engine()` in `douzero_ai.py` returns a `HybridAI` instance:
1. On init, tries to load DouZero ADP models (`douzero_models/douzero_landlord.ckpt`, etc.)
2. `decide()` uses DouZero if loaded (3-8ms inference on CPU), otherwise falls back to `DoudizhuAI` rule engine (~60ms)
3. DouZero models are ~6MB each, downloaded separately (not in repo)

### Model Registry

Model provenance is tracked in `models/MODEL_REGISTRY.json`, which records each model's version, training data, accuracy metrics, and source path. The active model is stored both in the registry and in `config.yaml` under `recognition.model_path`. Use `model_manager.py` to manage models — never modify these files manually.

## Common Commands

```bash
# Run the application (standard)
python main.py

# Run with path setup + module reload (use after editing modules)
python run_main.py

# Install dependencies
pip install -r requirements.txt

# Test AI engine logic (can_beat, decide, scoring)
python test_game.py

# Run specific AI test groups
python -c "import test_game; test_game.test_can_beat()"   # Core can_beat logic
python -c "import test_game; test_game.test_decide()"      # Decision logic
python -c "import test_game; test_game.run_simulation()"   # Full simulation
python -c "import test_game; test_game.test_triple_response()"   # Triple vs pair bug fix
python -c "import test_game; test_game.test_unknown_type_filter()"  # Unknown type fallback
python -c "import test_game; test_game.test_free_play_strategy()"   # First-play strategy
python -c "import test_game; test_game.test_chain_thinking()"       # Structure evaluation

# Test card detection on sample images
python test_detection.py

# Compare model detection results against ground truth
python test_comparison.py

# Test individual modules (each has a __main__ block)
python modules/state_manager.py
python modules/ai_engine.py
python modules/card_recognizer.py
python modules/window_capture.py
python modules/ui.py

# Model management
python model_manager.py list     # List all models
python model_manager.py use <name>  # Switch active model

# Iterative training (use iterative_train.py as main interface)
python iterative_train.py status    # Check training system status
python iterative_train.py train     # Start training (CPU, from v1)

# Dataset labeling tools (use labelme, not LabelImg)
start_labelme.bat               # Launch labelme for database images annotation
python convert_labelme_to_yolo.py  # Convert labelme JSON to YOLO TXT format
python auto_label.py            # Auto-generate YOLO labels from templates
python auto_label_v2.py         # Semi-auto labeling: YOLO detects positions + MiniMax reads card faces
python batch_labeler.py         # Batch label database images using YOLO detection only
python batch_labeler.py --verify   # Verify existing labels
python batch_labeler.py --limit 10 # Test with first 10 images

# Evaluate detection quality against ground truth labels
python evaluate_detection.py

# Verify label consistency
python check_labels.py

# Package as executable
pyinstaller --onefile --windowed --name DoudizhuAI main.py
```

## Multi-Agent Game System

```bash
# 多Agent自我对弈学习系统
python multi_agent_game.py learn [代数] [每代局数]  # 遗传算法学习
python multi_agent_game.py battle [局数]             # 对战测试
python multi_agent_game.py status                    # 查看学习状态

# 博弈模拟器
python game_referee.py    # 运行单局博弈模拟（直接加载模块，避免cv2递归）
python run_100_games.py   # 批量100场测试

# 策略对比测试
python strategy_tournament.py   # 简化策略对比（三种策略循环对战）

# DouZero集成（需要预训练模型）
python douzero_ai.py compare   # 与DouZero对比测试
```

### AI Strategy Test Results (100-game self-play tests)

| 地主配置 | 农民配置 | 地主胜率 |
|----------|----------|----------|
| balanced | aggressive + defensive | 100% |
| aggressive | balanced + balanced | 60% |
| defensive | aggressive + aggressive | 22% |

**最优配置**：地主用 `balanced`，农民用 `aggressive + defensive`

## Configuration (`config.yaml`)

```yaml
game:
  type: "欢乐斗地主"
  window_title: ".*斗地主.*"

recognition:
  fps: 1            # 截图帧率，1fps足够应付斗地主节奏
  confidence_threshold: 0.25  # YOLO 置信度阈值
  model_path: "models/yolov8_cards_v5.pt"  # 当前活跃模型

ai:
  strategy: "balanced"  # aggressive / balanced / defensive
  max_decision_time: 0.2

ui:
  opacity: 0.9
  position: "top-right"
  hotkey: "F1"

logging:
  level: "INFO"
  save_history: true
```

## Key Implementation Notes

### Thread Lifecycle
- UI 窗口启动后立即显示，但截图/处理线程阻塞于 `started=False`
- 用户点击"开始识别"后才真正开始处理
- 3线程：截图 → 处理 → UI（通过队列连接）

### Card Recognition Pipeline
- **区域裁剪是核心**：先裁剪到手牌区域再检测，YOLO 以接近训练分辨率的尺寸看到牌面
- **时序平滑**：5帧缓冲，60%多数投票，防止每帧结果跳动
- **识别稳定性**：只有数量或组成真正变化才更新结果
- **Rect平滑**：`get_window_rect()` 使用5帧移动平均消除 `win32gui.GetWindowRect` 的10-20px抖动
- `imgsz=480`（训练640），CPU上 ~300-600ms/帧；不要用960或1280

### Unknown Card Type Fallback
当对手出牌牌型识别失败时（`type=='unknown'`），用牌数量过滤而非跳过过滤，保证不会误推荐（如对手出三张时不会推荐对子）

### AI Strategy (`ai_engine.py`)

The AI evaluates every valid play through multiple weighted factors:
- **Efficiency**: Fewer cards played = better efficiency
- **Card type value**: Bombs (15) and rockets (20) score highest; singles/pairs score lower
- **Minimum-beat principle**: Prefer same-type plays that barely beat the last play
- **Chain thinking**: Evaluate how the remaining hand structure supports future plays
- **Role-aware**: Landlord plays low cards aggressively; farmers coordinate and intercept
- **Bomb reservation**: Save bombs for late game or critical moments
- **Safety net**: `decide()` always verifies the chosen play can beat `last_play`

**Beat hierarchy**: `rocket > bomb > all other plays` (must match type + length)

### UI Dragging
使用 Windows `WM_NCHITTEST` (`nativeEvent()`) 而非 Qt 鼠标事件处理拖拽。标题栏区域返回 `HTCAPTION` 让系统处理拖拽，避免干扰子控件点击。

### Window Missing Detection
游戏窗口消失后，3秒宽限期后推送 `{'window_missing': True}` 到 `result_queue`，UI 显示红色警告。

## Development Notes

- **Lazy cv2 import**：`card_recognizer.py` 和 `window_capture.py` 使用 `_get_cv2()` 函数内导入模式，避免 DLL/cv2 递归问题。禁止在模块级别 `import cv2`。
- **game_referee.py 直接导入**：使用 `importlib.util.spec_from_file_location()` 加载 `ai_engine` 和 `state_manager`，避免 PyQt5 通过 `modules/__init__.py` 触发 cv2 递归导入。
- **新增卡牌类型**：同时修改 `state_manager.py` 的 `_identify_card_type()` 和 `ai_engine.py` 的 `_generate_all_plays()`。
- **HybridAI 自动降级**：DouZero 模型加载失败时自动回退到 `DoudizhuAI`。
- `paddleocr>=2.7.0` 在 requirements.txt 中但未被使用，可移除。

## Training Pipeline

### Module Hierarchy

```
TrainingManager (high-level controller, `modules/training_manager.py`)
  └── IterativeTrainer (dataset YAML prep, model versioning, training execution, `modules/iterative_trainer.py`)
  └── ModelEvaluator (compare model accuracy, per-class F1, decide adoption, `modules/model_evaluator.py`)
```

Command-line entry point: `iterative_train.py` exposes CLI subcommands (status/label/train/history/compare).

### Scripts

- **`train_cpu.py`** — One-shot training script from v1 base. Uses fixed dataset (train + database), CPU-optimized params (batch=4, imgsz=640, epochs=30). Output: `models/yolov8_cards_v5.pt`. Not used for iterative runs.
- **`iterative_train.py`** — Main CLI for iterative training. Entry points: `python iterative_train.py status|label|train|history|compare`
- **`model_manager.py`** — Lightweight model switching and listing. Does **not** train; used to switch active model and inspect registry. Simpler alternative to `iterative_train.py status`.

Training data sourcing: `batch_labeler.py` sources images from the external directory `C:/Users/30330/Desktop/斗地主数据库`, labels them with YOLO-only detection, and copies images+labels to `dataset/images/database` and `dataset/labels/database`. The database directory is on the user's local desktop.

**标注工具优先级**: `labelme`（手动标注，最准确）> `auto_label_v2.py`（YOLO+MiniMax）> `batch_labeler.py`（纯YOLO）> `auto_label.py`（模板生成）

`python evaluate_detection.py` compares YOLO detection results against ground truth labels in `dataset/labels/all/`. Reports per-class precision, recall, F1, and lists error cases where F1 < 0.8.

## Model Management Skills

**模型处理规则**：
- ❌ 禁止删除任何模型文件
- ❌ 禁止替换已有模型（如 yolov8_cards.pt）
- ✅ 新模型只能添加新文件名（使用 `python model_manager.py add <path> [name]`）
- ✅ 可通过 `python model_manager.py list` 查看所有模型
- ✅ 可通过 `python model_manager.py use <name>` 切换活跃模型
- 模型文件存储在 `models/` 目录，所有历史版本都会被保留

## Training Data Management Skills

**数据保护规则**：
- ❌ 禁止删除 `dataset/images/train`、`dataset/labels/train`、`dataset/images/database`、`dataset/labels/database`、`dataset/candidate/` 下任何数据
- ✅ 训练必须使用全部数据（原始88张 + 数据库111张 + 候选数据）
- ✅ 新标注数据放入 `dataset/candidate/`
- ✅ 标签工具用 `labelme`（非LabelImg），转换用 `convert_labelme_to_yolo.py`

## 语言偏好

- 在解读信息、解释代码、回答问题等所有与用户的交流中，优先使用中文（简体）。
- 技术术语（如函数名、类名、变量名、库名）可以保留英文，但解释和上下文描述使用中文。

## Technology Stack

- **Window Capture**: pywin32 (win32gui) + mss
- **Card Detection**: YOLOv8 (15-class: 3-10, J, Q, K, A, 2, Joker_B, Joker_R) with CLAHE preprocessing
- **AI Decision**: Rule-based scoring engine with role-aware strategy
- **UI**: PyQt5 (frameless, always-on-top, translucent)
- **Platform**: Windows-only (win32gui dependency)
