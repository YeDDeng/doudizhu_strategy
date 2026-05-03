# 斗地主AI识别与决策系统 - 提示词手册

> 一套完整的AI驱动斗地主助手开发指南
> 版本: 1.0 | 更新日期: 2026-04-10

---

## 目录

1. [系统架构概览](#系统架构概览)
2. [模块1: 游戏窗口自动识别与捕获](#模块1-游戏窗口自动识别与捕获)
3. [模块2: 扑克牌图像识别引擎](#模块2-扑克牌图像识别引擎)
4. [模块3: 游戏状态管理与历史追踪](#模块3-游戏状态管理与历史追踪)
5. [模块4: AI决策引擎](#模块4-ai决策引擎)
6. [模块5: 用户界面与交互](#模块5-用户界面与交互)
7. [模块6: 主控制与集成](#模块6-主控制与集成)
8. [技术实现建议](#技术实现建议)
9. [项目文件结构](#项目文件结构)
10. [使用说明](#使用说明)

---

## 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    斗地主AI助手系统                          │
├─────────────────────────────────────────────────────────────┤
│  模块1: 窗口捕获  →  模块2: 图像识别  →  模块3: 状态管理    │
│      ↓                    ↓                    ↓            │
│  自动定位游戏窗口    识别牌面/数字/花色    记录出牌历史       │
│      ↓                    ↓                    ↓            │
│                    模块4: AI决策引擎  →  模块5: 界面展示    │
│                       智能出牌建议         实时悬浮窗提示     │
└─────────────────────────────────────────────────────────────┘
```

---

## 模块1: 游戏窗口自动识别与捕获

### 提示词

```markdown
# 角色：游戏窗口检测专家

## 任务
编写Python代码，自动检测并捕获斗地主游戏窗口的屏幕内容。

## 功能要求

### 1. 窗口检测
- 扫描所有运行中的窗口
- 通过窗口标题关键词识别斗地主游戏（支持：欢乐斗地主、腾讯斗地主、JJ斗地主等）
- 返回窗口句柄、位置和尺寸

### 2. 屏幕捕获
- 实时捕获游戏窗口区域
- 支持后台捕获（窗口被遮挡时也能抓取）
- 帧率要求：≥10 FPS

### 3. 区域定位
识别并标记以下关键区域：
- 我的手牌区域（底部）
- 上家出牌区域（顶部偏左）
- 下家出牌区域（顶部偏右）
- 地主标识区域
- 当前回合指示区域

## 输出格式
- Python代码（使用win32gui/mss/pygetwindow库）
- 区域坐标配置文件（相对坐标，适配不同分辨率）
- 错误处理机制（窗口未找到/关闭时的处理）

## 代码框架参考
```python
import win32gui
import win32con
import mss
import numpy as np

class WindowCapture:
    def __init__(self, window_title_keywords):
        self.keywords = window_title_keywords
        self.hwnd = None
        self.sct = mss.mss()
        
    def find_window(self):
        """查找斗地主游戏窗口"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(kw in title for kw in self.keywords):
                    self.hwnd = hwnd
                    return False
            return True
        win32gui.EnumWindows(callback, None)
        return self.hwnd
    
    def get_window_rect(self):
        """获取窗口位置和尺寸"""
        if not self.hwnd:
            return None
        return win32gui.GetWindowRect(self.hwnd)
    
    def capture(self):
        """捕获窗口截图"""
        rect = self.get_window_rect()
        if not rect:
            return None
        left, top, right, bottom = rect
        monitor = {"left": left, "top": top, "width": right-left, "height": bottom-top}
        return np.array(self.sct.grab(monitor))
```
```

---

## 模块2: 扑克牌图像识别引擎

### 提示词

```markdown
# 角色：计算机视觉+OCR识别专家

## 任务
构建高精度的扑克牌识别系统，能够从游戏截图中识别出牌面信息。

## 识别目标

| 类型 | 识别内容 | 难度 |
|------|----------|------|
| 我的手牌 | 12-20张牌，需识别数字+花色 | ⭐⭐⭐ |
| 已出牌（单张） | 单张牌清晰展示 | ⭐ |
| 已出牌（组合） | 顺子/连对/飞机等多张牌 | ⭐⭐⭐⭐ |
| 剩余牌数 | 对手手牌数量（数字显示） | ⭐ |

## 技术方案

### 1. 预处理
- 透视校正（处理斜拍角度）
- 色彩增强（突出红色/黑色牌面）
- 去噪和二值化

### 2. 识别方法（推荐多模型融合）

**方法A：YOLOv8目标检测 + CNN分类**
- YOLOv8定位每张牌的位置
- CNN分类器识别牌面数字和花色

**方法B：PaddleOCR**
- 直接识别数字和花色文字
- 适合清晰的牌面

**方法C：模板匹配**
- 预存54张牌的模板图
- 对特定游戏客户端效果好

### 3. 牌面编码规范
```
数字：3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2
花色：♠黑桃(Spade), ♥红心(Heart), ♣梅花(Club), ♦方块(Diamond)
特殊：小王(Joker_B), 大王(Joker_R)
```

## 输出格式

```python
{
    "my_cards": ["A♠", "K♥", "Q♣", "J♦", "10♠", "9♥", "8♣", "7♦", "6♠", "5♥", "4♣", "3♦"],
    "upper_player_last": ["J♠"],  # 上家最近出的牌
    "lower_player_last": ["Q♥", "Q♣", "Q♦"],  # 下家最近出的牌（三带一等）
    "upper_player_count": 15,  # 上家剩余牌数
    "lower_player_count": 12,  # 下家剩余牌数
    "landlord": "upper",  # 地主位置: upper/lower/self
    "current_turn": "self"  # 当前回合: upper/lower/self
}
```

## 精度要求
- 单张识别准确率 ≥ 98%
- 识别延迟 ≤ 100ms

## 代码框架参考

```python
from ultralytics import YOLO
import cv2
import numpy as np

class CardRecognizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.card_classes = ['3♠', '3♥', '3♣', '3♦', ..., 'Joker_R']
        
    def preprocess(self, image):
        """图像预处理"""
        # 调整大小、去噪、增强对比度
        img = cv2.resize(image, (1280, 720))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def detect_cards(self, image, region_type='my_hand'):
        """检测指定区域的牌"""
        results = self.model(image)
        cards = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                if conf > 0.8:
                    cards.append({
                        'card': self.card_classes[cls],
                        'confidence': float(conf),
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
        return cards
    
    def recognize(self, screenshot):
        """完整识别流程"""
        img = self.preprocess(screenshot)
        
        # 识别我的手牌
        my_cards = self.detect_cards(img, 'my_hand')
        
        # 识别上家出牌
        upper_cards = self.detect_cards(img, 'upper_play')
        
        # 识别下家出牌
        lower_cards = self.detect_cards(img, 'lower_play')
        
        # OCR识别剩余牌数
        upper_count = self.read_card_count(img, 'upper')
        lower_count = self.read_card_count(img, 'lower')
        
        return {
            'my_cards': [c['card'] for c in my_cards],
            'upper_player_last': [c['card'] for c in upper_cards],
            'lower_player_last': [c['card'] for c in lower_cards],
            'upper_player_count': upper_count,
            'lower_player_count': lower_count
        }
```
```

---

## 模块3: 游戏状态管理与历史追踪

### 提示词

```markdown
# 角色：游戏逻辑状态机工程师

## 任务
构建游戏状态管理系统，持续追踪牌局进展，为AI决策提供完整上下文。

## 核心功能

### 1. 牌库管理
- 初始化：54张牌，3人各17张，底牌3张
- 已出牌追踪（从识别结果中累加）
- 剩余牌推断（总牌数 - 已出 - 我的手牌）

### 2. 出牌历史记录
```python
history = [
    {"round": 1, "player": "upper", "cards": ["3♠"], "type": "single"},
    {"round": 2, "player": "lower", "cards": ["5♥", "5♣"], "type": "pair"},
    {"round": 3, "player": "self", "cards": ["A♠"], "type": "single"},
    # ...
]
```

### 3. 牌型识别
自动判断出牌类型：
- 单张 / 对子 / 三张 / 三带一 / 三带二
- 顺子（5张起）/ 连对（3对起）/ 飞机
- 炸弹（4张/王炸）
- 验证出牌合法性

### 4. 角色追踪
- 识别地主（通过界面标识或叫分过程）
- 判断盟友/对手关系

## 输出格式
- 状态机类设计（Python）
- 历史记录持久化（JSON/内存）
- 实时状态查询接口

## 代码框架参考

```python
from typing import List, Dict, Optional
from collections import Counter
import json

class GameStateManager:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.all_cards = self._init_deck()
        self.my_cards = []
        self.played_cards = []
        self.history = []
        self.current_round = 0
        self.landlord = None  # 'upper', 'lower', 'self'
        self.my_role = None   # 'landlord', 'farmer'
        self.upper_count = 17
        self.lower_count = 17
        self.last_play = None
        
    def _init_deck(self):
        """初始化一副牌"""
        suits = ['♠', '♥', '♣', '♦']
        ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
        deck = [r + s for s in suits for r in ranks]
        deck.extend(['Joker_B', 'Joker_R'])
        return deck
    
    def update_from_recognition(self, recognition_result: Dict):
        """从识别结果更新状态"""
        self.my_cards = recognition_result['my_cards']
        
        # 检测新出牌
        if recognition_result['upper_player_last']:
            self._record_play('upper', recognition_result['upper_player_last'])
        if recognition_result['lower_player_last']:
            self._record_play('lower', recognition_result['lower_player_last'])
        
        self.upper_count = recognition_result['upper_player_count']
        self.lower_count = recognition_result['lower_player_count']
    
    def _record_play(self, player: str, cards: List[str]):
        """记录一次出牌"""
        card_type = self._identify_card_type(cards)
        self.history.append({
            'round': self.current_round,
            'player': player,
            'cards': cards,
            'type': card_type,
            'timestamp': time.time()
        })
        self.played_cards.extend(cards)
        self.last_play = {'player': player, 'cards': cards, 'type': card_type}
        
        if player == 'upper':
            self.upper_count -= len(cards)
        elif player == 'lower':
            self.lower_count -= len(cards)
    
    def _identify_card_type(self, cards: List[str]) -> str:
        """识别牌型"""
        n = len(cards)
        if n == 0:
            return 'pass'
        if n == 1:
            return 'single'
        if n == 2:
            if 'Joker' in cards[0] and 'Joker' in cards[1]:
                return 'rocket'
            if self._get_rank(cards[0]) == self._get_rank(cards[1]):
                return 'pair'
        if n == 3:
            if self._same_rank(cards):
                return 'triple'
        if n == 4:
            if self._same_rank(cards):
                return 'bomb'
            # 三带一
            ranks = [self._get_rank(c) for c in cards]
            counter = Counter(ranks)
            if 3 in counter.values():
                return 'triple_with_single'
        # 顺子、连对等复杂牌型...
        return 'unknown'
    
    def _get_rank(self, card: str) -> str:
        """获取牌的点数"""
        if 'Joker' in card:
            return card
        return card[:-1]  # 去掉花色
    
    def _same_rank(self, cards: List[str]) -> bool:
        """判断是否同点数"""
        ranks = [self._get_rank(c) for c in cards]
        return len(set(ranks)) == 1
    
    def get_remaining_cards(self) -> List[str]:
        """获取剩余未出现的牌"""
        remaining = self.all_cards.copy()
        for card in self.my_cards + self.played_cards:
            if card in remaining:
                remaining.remove(card)
        return remaining
    
    def get_state(self) -> Dict:
        """获取当前完整状态"""
        return {
            'my_cards': self.my_cards,
            'my_role': self.my_role,
            'upper_cards_left': self.upper_count,
            'lower_cards_left': self.lower_count,
            'last_play': self.last_play,
            'history': self.history,
            'cards_seen': self.played_cards,
            'remaining_cards': self.get_remaining_cards()
        }
```
```

---

## 模块4: AI决策引擎

### 提示词

```markdown
# 角色：斗地主AI策略专家（强化学习+规则引擎）

## 任务
基于当前牌局状态，快速计算出最优出牌策略。

## 决策输入

```python
state = {
    "my_cards": ["2♠", "A♥", "K♣", "Q♦", "J♠", "10♥", "9♣", "8♦", "7♠", "6♥", "5♣", "4♦", "3♠", "3♥", "3♣"],
    "my_role": "farmer",  # farmer/landlord
    "upper_cards_left": 15,
    "lower_cards_left": 12,
    "last_play": {"player": "upper", "cards": ["J♠", "J♥", "J♣", "5♦"], "type": "triple_with_single"},
    "history": [...],
    "cards_seen": [...]  # 已出现的所有牌
}
```

## 决策策略（优先级排序）

### 第一层：必胜判断
- 检查是否一手出完（天牌）
- 检查是否有绝对压制的牌型

### 第二层：角色策略

**地主策略：**
- 优先出小牌，保留大牌控场
- 有炸弹时考虑拆牌逼炸
- 快速走完，不给农民配合机会

**农民策略（配合队友）：**
- 上家农民：顶牌（出大牌压制地主）
- 下家农民：顺牌（出小牌让队友过）
- 信号牌：特定牌型暗示手牌情况

### 第三层：牌型优化
- 计算最少手数出完的方案
- 优先出难配的组合（如单张多的顺子）
- 保留炸弹应对关键时刻

### 第四层：概率推算
- 推算对手可能持有的牌
- 基于剩余牌数评估风险
- 计算炸弹存在的概率

## 输出格式

```python
{
    "action": "play",  # play/pass
    "cards": ["3♠", "3♥", "3♣", "A♥"],  # 出的牌
    "type": "triple_with_single",  # 牌型
    "confidence": 0.85,  # 决策置信度
    "reasoning": "压制上家三带一，保留2和炸弹控场",  # 决策理由
    "alternatives": [...]  # 备选方案
}
```

## 性能要求
- 决策时间 ≤ 200ms
- 支持多线程并行计算

## 代码框架参考

```python
from typing import List, Dict, Tuple, Optional
from collections import Counter
import itertools
import random

class DoudizhuAI:
    # 牌值映射（用于比较大小）
    CARD_VALUE = {
        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15,
        'Joker_B': 16, 'Joker_R': 17
    }
    
    def __init__(self):
        self.strategy_mode = 'balanced'  # aggressive/balanced/defensive
    
    def decide(self, state: Dict) -> Dict:
        """主决策函数"""
        my_cards = state['my_cards']
        my_role = state['my_role']
        last_play = state['last_play']
        
        # 生成所有可能的出牌
        all_plays = self._generate_all_plays(my_cards)
        
        # 过滤合法出牌
        valid_plays = self._filter_valid_plays(all_plays, last_play)
        
        # 评估每个选项
        scored_plays = []
        for play in valid_plays:
            score = self._evaluate_play(play, state)
            scored_plays.append((play, score))
        
        # 选择最优
        if not scored_plays:
            return {'action': 'pass', 'reasoning': '无合法出牌'}
        
        scored_plays.sort(key=lambda x: x[1], reverse=True)
        best_play = scored_plays[0][0]
        
        return {
            'action': 'play',
            'cards': best_play['cards'],
            'type': best_play['type'],
            'confidence': scored_plays[0][1],
            'reasoning': self._generate_reasoning(best_play, state),
            'alternatives': [p[0] for p in scored_plays[1:3]]
        }
    
    def _generate_all_plays(self, cards: List[str]) -> List[Dict]:
        """生成所有可能的出牌组合"""
        plays = [{'cards': [], 'type': 'pass'}]  # 不出
        
        # 单张
        for card in set(cards):
            plays.append({'cards': [card], 'type': 'single'})
        
        # 对子
        rank_groups = self._group_by_rank(cards)
        for rank, group in rank_groups.items():
            if len(group) >= 2:
                plays.append({'cards': group[:2], 'type': 'pair'})
        
        # 三张
        for rank, group in rank_groups.items():
            if len(group) >= 3:
                plays.append({'cards': group[:3], 'type': 'triple'})
                # 三带一
                for other in set(cards) - set(group):
                    plays.append({'cards': group[:3] + [other], 'type': 'triple_with_single'})
        
        # 炸弹
        for rank, group in rank_groups.items():
            if len(group) == 4:
                plays.append({'cards': group, 'type': 'bomb'})
        
        # 王炸
        if 'Joker_B' in cards and 'Joker_R' in cards:
            plays.append({'cards': ['Joker_B', 'Joker_R'], 'type': 'rocket'})
        
        # 顺子（5张起）
        plays.extend(self._generate_straights(cards))
        
        # 连对
        plays.extend(self._generate_consecutive_pairs(cards))
        
        return plays
    
    def _group_by_rank(self, cards: List[str]) -> Dict:
        """按点数分组"""
        groups = {}
        for card in cards:
            rank = self._get_rank(card)
            if rank not in groups:
                groups[rank] = []
            groups[rank].append(card)
        return groups
    
    def _get_rank(self, card: str) -> str:
        """获取牌点数"""
        if 'Joker' in card:
            return card
        return card[:-1]
    
    def _filter_valid_plays(self, plays: List[Dict], last_play: Optional[Dict]) -> List[Dict]:
        """过滤合法出牌"""
        if not last_play or last_play['type'] == 'pass':
            return [p for p in plays if p['type'] != 'pass']
        
        valid = []
        for play in plays:
            if self._can_beat(play, last_play):
                valid.append(play)
        return valid
    
    def _can_beat(self, play: Dict, last_play: Dict) -> bool:
        """判断play是否能打过last_play"""
        # 王炸最大
        if play['type'] == 'rocket':
            return True
        if last_play['type'] == 'rocket':
            return False
        
        # 炸弹可以打任何非炸弹
        if play['type'] == 'bomb' and last_play['type'] != 'bomb':
            return True
        
        # 同类型比较
        if play['type'] != last_play['type']:
            return False
        
        # 比较大小
        play_value = self._get_play_value(play)
        last_value = self._get_play_value(last_play)
        return play_value > last_value
    
    def _get_play_value(self, play: Dict) -> int:
        """获取牌型的比较值"""
        if play['type'] == 'pass':
            return 0
        # 取最大的牌值
        return max(self.CARD_VALUE[self._get_rank(c)] for c in play['cards'])
    
    def _evaluate_play(self, play: Dict, state: Dict) -> float:
        """评估出牌得分"""
        score = 0.0
        my_cards = state['my_cards']
        my_role = state['my_role']
        
        # 1. 牌型价值
        if play['type'] == 'rocket':
            score += 100  # 王炸保留价值高
        elif play['type'] == 'bomb':
            score += 80
        
        # 2. 出牌效率（减少手牌数）
        cards_left = len(my_cards) - len(play['cards'])
        if cards_left == 0:
            score += 1000  # 出完获胜
        score += (17 - cards_left) * 2
        
        # 3. 角色策略
        if my_role == 'landlord':
            # 地主：优先出小牌
            if play['cards']:
                min_value = min(self.CARD_VALUE[self._get_rank(c)] for c in play['cards'])
                score += (20 - min_value) * 0.5
        else:
            # 农民：配合策略
            last_player = state.get('last_play', {}).get('player')
            if last_player == 'landlord':
                # 压制地主
                score += 10
        
        # 4. 保留大牌
        big_cards = ['2', 'A', 'K']
        for card in play['cards']:
            if self._get_rank(card) in big_cards:
                score -= 5
        
        # 5. 炸弹保留（关键时刻用）
        if play['type'] == 'bomb':
            # 对手牌少时考虑使用
            opponent_min = min(state['upper_cards_left'], state['lower_cards_left'])
            if opponent_min > 5:
                score -= 30  # 还早，保留
            else:
                score += 50  # 对手快出完，考虑使用
        
        return score
    
    def _generate_reasoning(self, play: Dict, state: Dict) -> str:
        """生成决策理由"""
        reasons = []
        
        if play['type'] == 'pass':
            return "选择不出，保留实力"
        
        if play['type'] in ['bomb', 'rocket']:
            reasons.append("使用炸弹压制")
        
        if state['my_role'] == 'farmer':
            reasons.append("农民配合策略")
        
        cards_left = len(state['my_cards']) - len(play['cards'])
        if cards_left <= 3:
            reasons.append(f"剩余{cards_left}张，即将出完")
        
        return "；".join(reasons) if reasons else "常规出牌"
    
    def _generate_straights(self, cards: List[str]) -> List[Dict]:
        """生成顺子组合"""
        plays = []
        ranks = sorted(set(self._get_rank(c) for c in cards if 'Joker' not in c),
                      key=lambda r: self.CARD_VALUE[r])
        
        # 找连续5张以上的序列
        for i in range(len(ranks)):
            for j in range(i + 4, min(i + 13, len(ranks))):
                sequence = ranks[i:j+1]
                if all(self.CARD_VALUE[sequence[k+1]] - self.CARD_VALUE[sequence[k]] == 1 
                       for k in range(len(sequence)-1)):
                    # 构建出牌
                    play_cards = []
                    for r in sequence:
                        for c in cards:
                            if self._get_rank(c) == r:
                                play_cards.append(c)
                                break
                    plays.append({'cards': play_cards, 'type': 'straight'})
        
        return plays
    
    def _generate_consecutive_pairs(self, cards: List[str]) -> List[Dict]:
        """生成连对组合"""
        plays = []
        rank_groups = self._group_by_rank(cards)
        pairs = [r for r, g in rank_groups.items() if len(g) >= 2 and 'Joker' not in r]
        pairs = sorted(pairs, key=lambda r: self.CARD_VALUE[r])
        
        # 找连续3对以上
        for i in range(len(pairs)):
            for j in range(i + 2, len(pairs)):
                sequence = pairs[i:j+1]
                if all(self.CARD_VALUE[sequence[k+1]] - self.CARD_VALUE[sequence[k]] == 1 
                       for k in range(len(sequence)-1)):
                    play_cards = []
                    for r in sequence:
                        play_cards.extend(rank_groups[r][:2])
                    plays.append({'cards': play_cards, 'type': 'consecutive_pairs'})
        
        return plays
```
```

---

## 模块5: 用户界面与交互

### 提示词

```markdown
# 角色：桌面应用UI设计师

## 任务
设计一个悬浮在游戏上方的AI助手界面，实时显示建议和操作按钮。

## 界面布局

```
┌─────────────────────────────────────┐
│  🤖 斗地主AI助手                    │
├─────────────────────────────────────┤
│  当前状态: 农民 | 剩余15张           │
├─────────────────────────────────────┤
│  💡 推荐出牌:                        │
│     三带一: 333+A                   │
│     理由: 压制对手，保留大牌         │
├─────────────────────────────────────┤
│  [♠] [♥] [♣] [♦] [2] [小王] [大王] │
│  剩余炸弹概率: 15%                   │
├─────────────────────────────────────┤
│  [一键出牌]  [pass]  [设置]          │
└─────────────────────────────────────┘
```

## 功能要求

### 1. 悬浮窗特性
- 置顶显示（always on top）
- 可拖动调整位置
- 透明度可调（不遮挡游戏）
- 自动吸附屏幕边缘

### 2. 实时显示
- 当前手牌（图标化展示）
- AI建议（高亮推荐）
- 对手剩余牌数
- 已推测出的对手可能牌型

### 3. 快捷操作
- 一键执行AI建议
- 手动选择出牌方案
- 暂停/恢复识别

### 4. 设置面板
- 游戏类型选择
- 识别灵敏度调节
- 快捷键绑定
- 日志记录开关

## 技术栈
- GUI框架：PyQt5/PySide6 或 Tkinter
- 样式：QSS/CSS美化
- 图标：扑克牌SVG图标集

## 代码框架参考

```python
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QApplication)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QColor, QPalette
import sys

class AIFloatingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_position = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 窗口设置
        self.setWindowFlags(
            Qt.FramelessWindowHint |  # 无边框
            Qt.WindowStaysOnTopHint |  # 置顶
            Qt.Tool  # 不在任务栏显示
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        
        # 主布局
        self.main_frame = QFrame(self)
        self.main_frame.setObjectName("mainFrame")
        self.main_frame.setStyleSheet("""
            #mainFrame {
                background-color: rgba(30, 30, 30, 220);
                border-radius: 10px;
                border: 2px solid #4CAF50;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        layout = QVBoxLayout(self.main_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题栏
        title_layout = QHBoxLayout()
        self.title_label = QLabel("🤖 斗地主AI助手")
        self.title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # 关闭按钮
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(25, 25)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff4444;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff4444;
                color: white;
            }
        """)
        self.close_btn.clicked.connect(self.hide)
        title_layout.addWidget(self.close_btn)
        layout.addLayout(title_layout)
        
        # 状态信息
        self.status_label = QLabel("当前状态: 等待游戏开始...")
        layout.addWidget(self.status_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555;")
        layout.addWidget(line)
        
        # AI建议区域
        suggestion_frame = QFrame()
        suggestion_layout = QVBoxLayout(suggestion_frame)
        suggestion_layout.setContentsMargins(0, 0, 0, 0)
        
        self.suggestion_title = QLabel("💡 推荐出牌:")
        self.suggestion_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        suggestion_layout.addWidget(self.suggestion_title)
        
        self.suggestion_cards = QLabel("等待识别...")
        self.suggestion_cards.setStyleSheet("color: #4CAF50; font-size: 14px;")
        suggestion_layout.addWidget(self.suggestion_cards)
        
        self.suggestion_reason = QLabel("")
        self.suggestion_reason.setStyleSheet("color: #aaa; font-size: 10px;")
        suggestion_layout.addWidget(self.suggestion_reason)
        
        layout.addWidget(suggestion_frame)
        
        # 手牌显示
        self.cards_label = QLabel("我的手牌: 识别中...")
        self.cards_label.setWordWrap(True)
        layout.addWidget(self.cards_label)
        
        # 对手信息
        self.opponent_label = QLabel("上家: ?张 | 下家: ?张")
        layout.addWidget(self.opponent_label)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("一键出牌")
        self.play_btn.setEnabled(False)
        btn_layout.addWidget(self.play_btn)
        
        self.pass_btn = QPushButton("不出")
        self.pass_btn.setEnabled(False)
        btn_layout.addWidget(self.pass_btn)
        
        self.settings_btn = QPushButton("设置")
        btn_layout.addWidget(self.settings_btn)
        
        layout.addLayout(btn_layout)
        
        # 设置主框架大小
        self.main_frame.setFixedSize(300, 350)
        self.setFixedSize(300, 350)
        
    def mousePressEvent(self, event):
        """鼠标按下事件（用于拖动）"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件（拖动窗口）"""
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def update_state(self, state: dict):
        """更新显示状态"""
        # 更新角色和手牌数
        role_text = "地主" if state.get('my_role') == 'landlord' else "农民"
        card_count = len(state.get('my_cards', []))
        self.status_label.setText(f"当前状态: {role_text} | 剩余{card_count}张")
        
        # 更新手牌显示
        cards = state.get('my_cards', [])
        self.cards_label.setText(f"我的手牌: {' '.join(cards)}")
        
        # 更新对手信息
        upper = state.get('upper_cards_left', '?')
        lower = state.get('lower_cards_left', '?')
        self.opponent_label.setText(f"上家: {upper}张 | 下家: {lower}张")
    
    def update_suggestion(self, decision: dict):
        """更新AI建议"""
        if decision['action'] == 'play':
            cards = ' '.join(decision['cards'])
            card_type = decision['type']
            self.suggestion_cards.setText(f"{card_type}: {cards}")
            self.suggestion_reason.setText(f"理由: {decision['reasoning']}")
            self.play_btn.setEnabled(True)
            self.pass_btn.setEnabled(True)
        else:
            self.suggestion_cards.setText("建议: 不出")
            self.suggestion_reason.setText(f"理由: {decision['reasoning']}")
            self.play_btn.setEnabled(False)
            self.pass_btn.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIFloatingWindow()
    window.show()
    sys.exit(app.exec_())
```
```

---

## 模块6: 主控制与集成

### 提示词

```markdown
# 角色：系统集成架构师

## 任务
编写主程序，整合所有模块，实现完整的AI助手工作流。

## 主循环流程

```python
while running:
    # 1. 捕获屏幕
    screenshot = capture_window(game_window)
    
    # 2. 识别牌面
    game_state = recognize_cards(screenshot)
    
    # 3. 更新状态机
    state_manager.update(game_state)
    
    # 4. AI决策
    if state_manager.my_turn:
        decision = ai_engine.decide(state_manager.get_state())
        ui.show_suggestion(decision)
    
    # 5. 更新UI
    ui.refresh(state_manager)
    
    time.sleep(0.1)  # 10 FPS
```

## 异常处理
- 游戏窗口关闭 → 暂停识别，等待重连
- 识别失败 → 使用上一次成功结果，标记置信度
- AI计算超时 → 使用规则引擎快速决策

## 配置系统

```yaml
game:
  type: "欢乐斗地主"  # 支持多种游戏客户端
  window_title: ".*斗地主.*"
  
recognition:
  fps: 10
  confidence_threshold: 0.8
  model_path: "models/yolov8_cards.pt"
  
ai:
  strategy: "balanced"  # aggressive/balanced/defensive
  max_decision_time: 0.2
  
ui:
  opacity: 0.9
  position: "top-right"
  hotkey: "F1"
  
logging:
  level: "INFO"
  save_history: true
```

## 代码框架参考

```python
import yaml
import time
import threading
from queue import Queue
import logging

# 导入各模块
from modules.window_capture import WindowCapture
from modules.card_recognizer import CardRecognizer
from modules.state_manager import GameStateManager
from modules.ai_engine import DoudizhuAI
from modules.ui import AIFloatingWindow

class DoudizhuAssistant:
    def __init__(self, config_path='config.yaml'):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化日志
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化各模块
        self.window_capture = WindowCapture(
            self.config['game']['window_title']
        )
        self.card_recognizer = CardRecognizer(
            self.config['recognition']['model_path']
        )
        self.state_manager = GameStateManager()
        self.ai_engine = DoudizhuAI()
        
        # UI在主线程运行
        self.ui = None
        
        # 控制标志
        self.running = False
        self.paused = False
        
        # 数据队列
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        
    def start(self):
        """启动助手"""
        self.logger.info("斗地主AI助手启动中...")
        
        # 查找游戏窗口
        if not self.window_capture.find_window():
            self.logger.error("未找到斗地主游戏窗口，请确保游戏已运行")
            return False
        
        self.logger.info("已找到游戏窗口")
        
        # 启动后台处理线程
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.process_thread = threading.Thread(target=self._process_loop)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        # 启动UI（主线程）
        self._run_ui()
        
        return True
    
    def stop(self):
        """停止助手"""
        self.logger.info("正在停止...")
        self.running = False
        self.capture_thread.join()
        self.process_thread.join()
    
    def _capture_loop(self):
        """屏幕捕获循环"""
        fps = self.config['recognition']['fps']
        interval = 1.0 / fps
        
        while self.running:
            if self.paused:
                time.sleep(interval)
                continue
            
            try:
                # 捕获屏幕
                screenshot = self.window_capture.capture()
                if screenshot is None:
                    self.logger.warning("捕获失败，尝试重新查找窗口")
                    if not self.window_capture.find_window():
                        time.sleep(1)
                        continue
                
                # 放入队列（丢弃旧帧）
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(screenshot)
                
            except Exception as e:
                self.logger.error(f"捕获异常: {e}")
            
            time.sleep(interval)
    
    def _process_loop(self):
        """图像处理和AI决策循环"""
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # 获取最新帧
                if self.frame_queue.empty():
                    time.sleep(0.05)
                    continue
                
                screenshot = self.frame_queue.get()
                
                # 识别牌面
                recognition_result = self.card_recognizer.recognize(screenshot)
                
                # 更新状态
                self.state_manager.update_from_recognition(recognition_result)
                
                # AI决策（如果到我出牌）
                if recognition_result.get('current_turn') == 'self':
                    state = self.state_manager.get_state()
                    decision = self.ai_engine.decide(state)
                    
                    # 发送到UI
                    if self.ui:
                        self.ui.update_state(state)
                        self.ui.update_suggestion(decision)
                
            except Exception as e:
                self.logger.error(f"处理异常: {e}")
    
    def _run_ui(self):
        """运行UI（主线程）"""
        from PyQt5.QtWidgets import QApplication
        import sys
        
        app = QApplication(sys.argv)
        self.ui = AIFloatingWindow()
        
        # 绑定按钮事件
        self.ui.play_btn.clicked.connect(self._on_play_clicked)
        self.ui.pass_btn.clicked.connect(self._on_pass_clicked)
        self.ui.settings_btn.clicked.connect(self._on_settings_clicked)
        
        self.ui.show()
        sys.exit(app.exec_())
    
    def _on_play_clicked(self):
        """一键出牌按钮"""
        self.logger.info("执行AI建议出牌")
        # 这里可以实现自动点击游戏出牌按钮的功能
        # 需要配合自动化工具如pyautogui
    
    def _on_pass_clicked(self):
        """不出按钮"""
        self.logger.info("选择不出")
    
    def _on_settings_clicked(self):
        """设置按钮"""
        self.logger.info("打开设置")


def main():
    assistant = DoudizhuAssistant()
    assistant.start()


if __name__ == '__main__':
    main()
```
```

---

## 技术实现建议

| 模块 | 推荐技术栈 | 备选方案 |
|------|-----------|----------|
| 窗口捕获 | `pygetwindow` + `mss` | `PIL.ImageGrab` |
| 图像识别 | `YOLOv8` + `PaddleOCR` | `OpenCV`模板匹配 |
| AI决策 | `Python`规则引擎 | `PyTorch`强化学习 |
| UI界面 | `PyQt5`/`PySide6` | `DearPyGui` |
| 打包发布 | `PyInstaller` | `Nuitka` |

### 依赖列表（requirements.txt）

```
pygetwindow>=0.0.9
mss>=9.0.1
opencv-python>=4.8.0
ultralytics>=8.0.0
paddleocr>=2.7.0
PyQt5>=5.15.0
pyyaml>=6.0
numpy>=1.24.0
pillow>=10.0.0
pywin32>=306; platform_system=="Windows"
```

---

## 项目文件结构

```
ddz-ai-assistant/
├── main.py                 # 主程序入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── README.md               # 项目说明
├── modules/
│   ├── __init__.py
│   ├── window_capture.py   # 窗口捕获
│   ├── card_recognizer.py  # 牌面识别
│   ├── state_manager.py    # 状态管理
│   ├── ai_engine.py        # AI决策
│   └── ui.py               # 用户界面
├── models/
│   ├── yolov8_cards.pt     # 训练好的检测模型
│   └── card_templates/     # 牌面模板图
├── assets/
│   └── card_icons/         # 扑克牌图标
├── tests/
│   └── test_images/        # 测试截图
└── docs/
    └── api.md              # API文档
```

---

## 使用说明

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourname/ddz-ai-assistant.git
cd ddz-ai-assistant

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

编辑 `config.yaml`：
- 设置游戏窗口标题关键词
- 调整识别灵敏度
- 配置AI策略模式

### 3. 运行程序

```bash
python main.py
```

### 4. 使用流程

1. 启动斗地主游戏客户端
2. 运行AI助手程序
3. 程序自动检测游戏窗口
4. 进入对局后，AI助手开始识别和提示
5. 根据AI建议进行出牌决策

---

## 注意事项

1. **合规性**：本工具仅供学习研究使用，请勿用于违规场景
2. **识别精度**：首次使用建议在训练模式下校准识别模型
3. **性能要求**：建议CPU i5以上，内存8G以上
4. **分辨率**：支持多种分辨率，但推荐使用1920x1080获得最佳效果

---

## 后续优化方向

1. 引入强化学习训练更强大的AI
2. 支持更多斗地主游戏客户端
3. 添加出牌自动化功能（需配合模拟点击）
4. 云端模型更新，持续提升识别准确率
5. 添加语音播报功能

---

*文档版本: 1.0 | 最后更新: 2026-04-10 *
