"""
DouZero AI 集成模块
提供实时推理引擎，将DouZero模型直接集成到AI助手中。

推理速度：CPU上约8ms/次决策

模型位置：douzero_models/
  - douzero_landlord.ckpt    (地主)
  - douzero_landlord_up.ckpt   (农民-地主上家)
  - douzero_landlord_down.ckpt (农民-地主下家)
"""

import os
import sys
import time
from collections import Counter
import numpy as np

# DouZero相关导入
try:
    from douzero.evaluation.deep_agent import DeepAgent
    from douzero.evaluation.deep_agent import _load_model
    from douzero.env.env import get_obs
    from douzero.env.move_generator import MovesGener
    from douzero.env.game import InfoSet
    import torch

    DOUZERO_AVAILABLE = True
except ImportError as e:
    print(f"[DouZero] Import failed: {e}")
    DOUZERO_AVAILABLE = False

# 模型路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), "douzero_models")
MODEL_PATHS = {
    'landlord': os.path.join(MODEL_DIR, "douzero_landlord.ckpt"),
    'landlord_up': os.path.join(MODEL_DIR, "douzero_landlord_up.ckpt"),
    'landlord_down': os.path.join(MODEL_DIR, "douzero_landlord_down.ckpt"),
}

# 牌面值 → DouZero值映射（与ai_engine._card_to_env一致）
RANK_TO_VALUE = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17,
    'Joker_B': 20, 'Joker_R': 30, 'JB': 20, 'JR': 30,
}
VALUE_TO_RANK = {v: k for k, v in RANK_TO_VALUE.items()}
# 54张完整牌组的DouZero值
FULL_DECK_VALUES = [3]*4 + [4]*4 + [5]*4 + [6]*4 + [7]*4 + [8]*4 + [9]*4 + [10]*4 + \
                   [11]*4 + [12]*4 + [13]*4 + [14]*4 + [17]*4 + [20]*1 + [30]*1


def card_to_value(card):
    """将我们的牌（如'3S', 'JB', 'Joker_R'）转为DouZero环境值"""
    if card.startswith('Joker') or card in ('JB', 'JR'):
        return 20 if card in ('Joker_B', 'JB') else 30
    if len(card) >= 2 and card[1] == '0':
        return 10
    return RANK_TO_VALUE.get(card[0], 0)


def douzero_values_from_hand(cards):
    """将我们的手牌转为DouZero值列表（丢弃花色）"""
    return sorted([card_to_value(c) for c in cards])


def _get_rank(card):
    """提取牌的面值文字"""
    if card.startswith('Joker') or card in ('JB', 'JR'):
        return 'Joker_B' if card in ('Joker_B', 'JB') else 'Joker_R'
    if len(card) >= 2 and card[1] == '0':
        return '10'
    return card[0]


def match_action_to_cards(action_values, our_cards):
    """将DouZero动作（值列表）匹配到我们手牌的实际牌。

    例如 action=[3,3,3] 从 ['3S','3H','3D','3C','5D',...] 中选出 ['3S','3H','3D']
    """
    if not action_values:
        return []
    result = []
    remaining = list(our_cards)
    for val in action_values:
        # 找一张匹配该值的手牌
        for c in remaining:
            if card_to_value(c) == val:
                result.append(c)
                remaining.remove(c)
                break
        else:
            # 理论上不会到这里（动作应该合法）
            pass
    return result


def build_legal_actions(our_values):
    """生成所有合法出牌（DouZero格式的值列表）"""
    mg = MovesGener(our_values)
    return mg.gen_moves()


def _is_consecutive(vals):
    """判断一组值是否连续递增（排除2=17和王）"""
    vals = sorted(set(v for v in vals if v < 17))
    if len(vals) < 2:
        return False
    return vals[-1] - vals[0] == len(vals) - 1


def filter_legal_by_last_play(legal_actions, last_play_values, last_play_type):
    """过滤出能打过上家的合法动作。
    使用 last_play_type 字符串判断牌型，而非比较去重集合。
    返回 (filtered_actions, has_same_type)
    """
    if not last_play_values:
        return legal_actions, True

    last_len = len(last_play_values)
    last_counts = Counter(last_play_values)
    last_max = max(last_play_values)
    last_is_rocket = (20 in last_play_values and 30 in last_play_values)
    last_is_bomb = not last_is_rocket and last_len == 4 and len(last_counts) == 1
    last_mc = last_counts.most_common()

    valid = []
    has_same_type = False

    for action in legal_actions:
        a_counts = Counter(action)
        a_len = len(action)
        a_mc = a_counts.most_common()

        # 王炸 > 所有（除王炸本身）
        if 20 in action and 30 in action:
            if not last_is_rocket:
                valid.append(action)
            continue

        # 炸弹 > 非火箭
        if a_len == 4 and len(a_counts) == 1 and not last_is_rocket:
            if last_is_bomb and max(action) <= last_max:
                continue
            valid.append(action)
            continue

        # 非炸弹/火箭不能打炸弹/火箭
        if last_is_bomb or last_is_rocket:
            continue

        # === 以下按牌型分别比较 ===
        if last_play_type == 'single':
            if a_len == 1 and max(action) > last_max:
                has_same_type = True
                valid.append(action)

        elif last_play_type == 'pair':
            if a_len == 2 and len(a_counts) == 1 and max(action) > last_max:
                has_same_type = True
                valid.append(action)

        elif last_play_type == 'triple':
            if a_len == 3 and len(a_counts) == 1 and max(action) > last_max:
                has_same_type = True
                valid.append(action)

        elif last_play_type == 'triple_with_single':
            # 4张牌，3+1分布
            if a_len == 4 and len(a_counts) == 2:
                if a_mc[0][1] == 3 and a_mc[1][1] == 1:
                    a_triple_val = a_mc[0][0]
                    last_triple_val = last_mc[0][0]  # last也是3+1分布
                    if a_triple_val > last_triple_val:
                        has_same_type = True
                        valid.append(action)

        elif last_play_type == 'triple_with_pair':
            # 5张牌，3+2分布
            if a_len == 5 and len(a_counts) == 2:
                if sorted(a_counts.values()) == [2, 3]:
                    a_triple_val = a_mc[0][0]
                    last_triple_val = last_mc[0][0]
                    if a_triple_val > last_triple_val:
                        has_same_type = True
                        valid.append(action)

        elif last_play_type == 'straight':
            # 顺子：每值1张，连续，不含2/王
            if a_len == last_len and len(a_counts) == a_len:
                a_vals = sorted(a_counts.keys())
                if all(v < 17 for v in a_vals):
                    if a_vals[-1] - a_vals[0] == a_len - 1:
                        if a_vals[-1] > last_max:
                            has_same_type = True
                            valid.append(action)

        elif last_play_type == 'consecutive_pairs':
            # 连对：长度偶数，每值2张，连续，不含2/王
            if a_len == last_len and a_len % 2 == 0:
                pairs_n = a_len // 2
                if len(a_counts) == pairs_n and all(c == 2 for c in a_counts.values()):
                    a_vals = sorted(a_counts.keys())
                    if all(v < 17 for v in a_vals):
                        if a_vals[-1] - a_vals[0] == pairs_n - 1:
                            if a_vals[-1] > max(last_counts.keys()):
                                has_same_type = True
                                valid.append(action)

        elif last_play_type == 'airplane':
            # 飞机不带：连续三张，每值3张，无翅膀
            a_triples = sorted([v for v in a_counts if a_counts[v] >= 3])
            last_triples = sorted([v for v in last_counts if last_counts[v] >= 3])
            if len(a_triples) == len(last_triples) and a_len == last_len:
                if all(v < 17 for v in a_triples) and _is_consecutive(a_triples):
                    if a_triples[-1] > last_triples[-1]:
                        has_same_type = True
                        valid.append(action)

        elif last_play_type == 'airplane_with_singles':
            # 飞机带单：连续三张 + 每三张带一单
            a_triples = sorted([v for v in a_counts if a_counts[v] >= 3])
            last_triples = sorted([v for v in last_counts if last_counts[v] >= 3])
            if len(a_triples) == len(last_triples):
                triple_n = len(a_triples)
                if a_len == triple_n * 4 and last_len == triple_n * 4:
                    if all(v < 17 for v in a_triples) and _is_consecutive(a_triples):
                        if a_triples[-1] > last_triples[-1]:
                            has_same_type = True
                            valid.append(action)

        elif last_play_type == 'airplane_with_pairs':
            a_triples = sorted([v for v in a_counts if a_counts[v] >= 3])
            last_triples = sorted([v for v in last_counts if last_counts[v] >= 3])
            if len(a_triples) == len(last_triples):
                triple_n = len(a_triples)
                if a_len == triple_n * 5 and last_len == triple_n * 5:
                    if all(v < 17 for v in a_triples) and _is_consecutive(a_triples):
                        if a_triples[-1] > last_triples[-1]:
                            has_same_type = True
                            valid.append(action)

        elif last_play_type == 'four_with_two_singles':
            if a_len == 6 and len(a_counts) == 3:
                a_quad = [v for v in a_counts if a_counts[v] == 4]
                last_quad = [v for v in last_counts if last_counts[v] == 4]
                if a_quad and last_quad and a_quad[0] > last_quad[0]:
                    has_same_type = True
                    valid.append(action)

        elif last_play_type == 'four_with_two_pairs':
            if a_len == 8 and len(a_counts) >= 2:
                a_quad = [v for v in a_counts if a_counts[v] == 4]
                last_quad = [v for v in last_counts if last_counts[v] == 4]
                if a_quad and last_quad and a_quad[0] > last_quad[0]:
                    has_same_type = True
                    valid.append(action)

        elif last_play_type == 'unknown':
            # 未知类型：用长度 + 值比较（兜底）
            if a_len == last_len and max(action) > last_max:
                has_same_type = True
                valid.append(action)

    return valid, has_same_type


class DouZeroRealTime:
    """DouZero实时推理引擎。

    直接调用DouZero模型前向推理，不走完整GameEnv。
    CPU上约8ms/次，适合实时出牌建议。
    """

    def __init__(self):
        self.models = {}       # position → model
        self.loaded = False
        self.load_time = 0.0

    def load_models(self, positions=None):
        """加载模型。positions=None加载全部三个"""
        if not DOUZERO_AVAILABLE:
            return False
        if positions is None:
            positions = ['landlord', 'landlord_up', 'landlord_down']
        t0 = time.time()
        for pos in positions:
            path = MODEL_PATHS.get(pos)
            if path and os.path.exists(path):
                try:
                    self.models[pos] = _load_model(pos, path)
                except Exception as e:
                    print(f"[DouZero] Failed to load {pos}: {e}")
        self.load_time = time.time() - t0
        self.loaded = len(self.models) > 0
        if self.loaded:
            self.model = list(self.models.values())[0]  # 默认用最后一个加载的
        return self.loaded

    def _build_infoset(self, state, my_role):
        """从我们的state构建DouZero InfoSet"""
        my_cards = state.get('my_cards', [])
        my_values = douzero_values_from_hand(my_cards)
        last_play = state.get('last_play')
        play_history = state.get('play_history', [])

        # 确定DouZero位置
        if my_role == 'landlord':
            position = 'landlord'
        else:
            # 农民：简单用landlord_down（不影响出牌质量）
            position = 'landlord_down'

        # 构建InfoSet
        infoset = InfoSet(position)
        infoset.player_hand_cards = my_values

        # 估算其他玩家手牌（全量 - 我的手牌 - 已出的牌）
        played_values = []
        for h in play_history:
            for c in h.get('cards', []):
                played_values.append(card_to_value(c))
        # 如果last_play也存在但不在history中
        if last_play and last_play.get('cards'):
            for c in last_play['cards']:
                v = card_to_value(c)
                if v not in played_values:
                    played_values.append(v)

        remaining = list(FULL_DECK_VALUES)
        for v in my_values:
            if v in remaining:
                remaining.remove(v)
        for v in played_values:
            if v in remaining:
                remaining.remove(v)
        infoset.other_hand_cards = remaining

        # 手牌数量
        upper_count = state.get('upper_player_count', 17)
        lower_count = state.get('lower_player_count', 17)
        if position == 'landlord':
            infoset.num_cards_left_dict = {
                'landlord': len(my_values),
                'landlord_up': upper_count,
                'landlord_down': lower_count
            }
        elif position == 'landlord_down':
            infoset.num_cards_left_dict = {
                'landlord': upper_count,
                'landlord_up': lower_count,
                'landlord_down': len(my_values)
            }
        else:  # landlord_up
            infoset.num_cards_left_dict = {
                'landlord': lower_count,
                'landlord_up': len(my_values),
                'landlord_down': upper_count
            }

        infoset.three_landlord_cards = []

        # 最近出牌
        if last_play and last_play.get('cards'):
            last_values = [card_to_value(c) for c in last_play['cards']]
            infoset.last_move = last_values
            infoset.last_two_moves = [last_values, []]
        else:
            infoset.last_move = []
            infoset.last_two_moves = [[], []]

        # last_move_dict
        infoset.last_move_dict = {
            'landlord': [],
            'landlord_up': [],
            'landlord_down': []
        }

        # 出牌序列
        seq = []
        for h in play_history:
            seq.append([card_to_value(c) for c in h.get('cards', [])])
        if last_play and last_play.get('cards'):
            lv = [card_to_value(c) for c in last_play['cards']]
            if not seq or seq[-1] != lv:
                seq.append(lv)
        infoset.card_play_action_seq = seq

        # 已出牌
        infoset.played_cards = {
            'landlord': [],
            'landlord_up': [],
            'landlord_down': []
        }

        # 全手牌
        infoset.all_handcards = {
            'landlord': [],
            'landlord_up': [],
            'landlord_down': []
        }

        infoset.bomb_num = state.get('bomb_num', 0)
        infoset.last_pid = 'landlord'

        # 生成合法出牌
        all_actions = build_legal_actions(my_values)

        # 过滤出能管上上家的
        last_play_values = []
        last_play_type = None
        if last_play and last_play.get('cards'):
            last_play_values = [card_to_value(c) for c in last_play['cards']]
            last_play_type = last_play.get('type')

        if last_play_values and last_play_type != 'pass':
            filtered, has_same = filter_legal_by_last_play(all_actions, last_play_values, last_play_type)
            infoset.legal_actions = filtered
        else:
            infoset.legal_actions = all_actions

        return infoset, position

    def decide(self, state):
        """根据state做出决策，返回与ai_engine兼容的dict"""
        my_cards = state.get('my_cards', [])
        my_role = state.get('my_role', 'farmer')
        last_play = state.get('last_play')

        if not my_cards:
            return {'action': 'pass', 'cards': [], 'type': 'pass', 'confidence': 1.0,
                    'reasoning': 'No cards to play', 'alternatives': []}

        # 确定使用哪个模型
        if my_role == 'landlord':
            position = 'landlord'
        else:
            position = 'landlord_down'

        model = self.models.get(position)
        if model is None:
            # 回退到第一个加载的模型
            if self.models:
                model = list(self.models.values())[0]
            else:
                return {'action': 'pass', 'cards': [], 'type': 'pass', 'confidence': 0,
                        'reasoning': 'No DouZero model loaded', 'alternatives': []}

        # 构建InfoSet
        infoset, _ = self._build_infoset(state, my_role)

        # 如果只有一个合法动作
        if len(infoset.legal_actions) == 0:
            return {'action': 'pass', 'cards': [], 'type': 'pass', 'confidence': 1.0,
                    'reasoning': 'Cannot beat last play', 'alternatives': []}
        if len(infoset.legal_actions) == 1:
            action = infoset.legal_actions[0]
        else:
            # 运行模型前向推理
            obs = get_obs(infoset)
            z_batch = torch.from_numpy(obs['z_batch']).float()
            x_batch = torch.from_numpy(obs['x_batch']).float()
            with torch.no_grad():
                y_pred = model.forward(z_batch, x_batch, return_value=True)['values']
            y_pred = y_pred.detach().cpu().numpy()
            best_idx = int(np.argmax(y_pred, axis=0)[0])
            action = infoset.legal_actions[best_idx]

        # 将DouZero动作转回我们的牌
        played_cards = match_action_to_cards(action, my_cards)

        # 判断牌型（复用ai_engine的逻辑太麻烦，简单判断）
        play_type = self._classify_action(action)

        return {
            'action': 'play' if played_cards else 'pass',
            'cards': played_cards,
            'type': play_type,
            'confidence': 0.9,
            'reasoning': f'DouZero建议: {play_type}',
            'alternatives': []
        }

    def _classify_action(self, action):
        """简单判断DouZero动作的牌型"""
        if not action:
            return 'pass'
        n = len(action)
        s = sorted(set(action))
        if n == 1:
            return 'single'
        if n == 2:
            if 20 in action and 30 in action:
                return 'rocket'
            if len(s) == 1:
                return 'pair'
            return 'single'  # 不连续单张（实际上不是合法出牌，但兜底）
        if n == 3 and len(s) == 1:
            return 'triple'
        if n == 4:
            if len(s) == 1:
                return 'bomb'
            if Counter(action).most_common(1)[0][1] == 3:
                return 'triple_with_single'
            return 'unknown'
        if n == 5:
            if Counter(action).most_common(1)[0][1] == 3:
                return 'triple_with_pair'
            # 可能顺子
            if self._is_consecutive(s) and len(s) == 5:
                return 'straight'
            return 'unknown'
        # 更长的牌
        if self._is_consecutive(s) and n == len(s):
            return 'straight'
        # 连对
        if n % 2 == 0 and n >= 6:
            pairs = all(Counter(action)[v] >= 2 for v in s)
            if pairs and self._is_consecutive(s) and n // 2 == len(s):
                return 'consecutive_pairs'
        # 飞机
        triples = [v for v in s if Counter(action)[v] >= 3]
        if len(triples) >= 2 and self._is_consecutive(triples):
            triple_count = sum(Counter(action)[v] // 3 for v in triples)
            if n == triple_count * 3:
                return 'airplane'
            if n == triple_count * 4:
                return 'airplane_with_singles'
            if n == triple_count * 5:
                return 'airplane_with_pairs'
        if n == 6:
            c = Counter(action)
            if len(s) == 3 and c.most_common(1)[0][1] == 4:
                return 'four_with_two_singles'
        if n == 8:
            c = Counter(action)
            if c.most_common(1)[0][1] == 4:
                # 剩下的4张组成两对？
                remaining = [v for v in s if c[v] < 4]
                if len(remaining) <= 2:
                    return 'four_with_two_pairs'
        return 'unknown'

    def _is_consecutive(self, vals):
        """判断值列表是否连续（排除2和王的连续检查）"""
        vals = sorted(v for v in vals if v < 17)
        if len(vals) < 2:
            return False
        return vals[-1] - vals[0] == len(vals) - 1


# === 混合引擎：DouZero + 规则AI ===

class HybridAI:
    """混合AI引擎。DouZero可用时用它，否则回退规则AI。decide()接口完全兼容。"""

    def __init__(self, strategy_mode='balanced'):
        self.strategy_mode = strategy_mode
        self._rule_ai = None
        self._douzero = None
        self.use_douzero = False
        self._init_douzero()

    def _init_douzero(self):
        if not DOUZERO_AVAILABLE:
            return
        engine = DouZeroRealTime()
        if engine.load_models():
            self._douzero = engine
            self.use_douzero = True
            print(f"[HybridAI] DouZero引擎已加载 ({engine.load_time:.1f}s)")

    def _get_rule_ai(self):
        if self._rule_ai is None:
            from ai_engine import DoudizhuAI
            self._rule_ai = DoudizhuAI(self.strategy_mode)
        return self._rule_ai

    def decide(self, state):
        if self.use_douzero and self._douzero:
            try:
                return self._douzero.decide(state)
            except Exception as e:
                print(f"[HybridAI] DouZero错误: {e}, 回退规则AI")
                self.use_douzero = False
        return self._get_rule_ai().decide(state)


def create_ai_engine(strategy_mode='balanced'):
    """工厂函数：DouZero可用时返回HybridAI，否则DoudizhuAI。
    在main.py中替换 `DoudizhuAI(strategy)` 调用即可。
    """
    if DOUZERO_AVAILABLE:
        engine = HybridAI(strategy_mode)
        if engine.use_douzero:
            return engine
    from ai_engine import DoudizhuAI
    return DoudizhuAI(strategy_mode)


# === 快速测试 ===
if __name__ == "__main__":
    # 测速度
    engine = DouZeroRealTime()
    ok = engine.load_models()
    if not ok:
        print("无法加载模型")
        sys.exit(1)

    test_hand = ['3S', '3H', '3D', '4S', '5S', '6S', '7S', '8S', '9S', '10S', 'JS', '2S', 'JB', 'JR']

    # 自由出牌测试
    state = {
        'my_cards': test_hand,
        'my_role': 'landlord',
        'last_play': None,
        'upper_player_count': 17,
        'lower_player_count': 17,
        'play_history': []
    }
    t0 = time.perf_counter()
    dec = engine.decide(state)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"自由出牌: {dec['type']} ({len(dec['cards'])}张) 耗时{elapsed:.1f}ms")

    # 响应出牌测试
    state2 = {
        'my_cards': test_hand,
        'my_role': 'landlord',
        'last_play': {'player': 'upper', 'type': 'pair', 'cards': ['KS', 'KH'], 'value': 13},
        'upper_player_count': 16,
        'lower_player_count': 16,
        'play_history': [{'player': 'upper', 'type': 'pair', 'cards': ['KS', 'KH'], 'value': 13}]
    }
    t0 = time.perf_counter()
    dec2 = engine.decide(state2)
    elapsed2 = (time.perf_counter() - t0) * 1000
    print(f"响应出牌: {dec2['type']} ({len(dec2['cards'])}张) 耗时{elapsed2:.1f}ms")

    # 批量测速
    N = 50
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        engine.decide(state)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\n{N}次平均决策时间: {sum(times)/N:.1f}ms")
    print(f"  最小值: {min(times):.1f}ms")
    print(f"  最大值: {max(times):.1f}ms")
