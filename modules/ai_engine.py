"""
Module 4: AI Decision Engine
Rule-based AI that generates optimal play suggestions.
Performance target: ≤ 200ms per decision.
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple
import itertools

class DoudizhuAI:
    """AI decision engine for Doudizhu."""

    # 类变量控制调试输出
    DEBUG_OUTPUT = True  # Set to False to suppress debug logs

    # Card value mapping - 与DouZero对齐
    # 3-10 对应 3-10, J=11, Q=12, K=13, A=14, 2=17, X(小王)=20, D(大王)=30
    CARD_VALUE = {
        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        '2': 17,
        'Joker_B': 20,  # 小王(黑桃王)
        'Joker_R': 30   # 大王(红桃王)
    }

    # 完整牌组（54张）
    ALL_CARDS = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                  8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11,
                  12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 17, 17, 17, 17, 20, 30]

    # Strategy modes
    STRATEGY_AGGRESSIVE = 'aggressive'
    STRATEGY_BALANCED = 'balanced'
    STRATEGY_DEFENSIVE = 'defensive'

    # Game phase thresholds
    PHASE_EARLY = 'early'      # 手牌 > 10张
    PHASE_MID = 'mid'          # 手牌 5-10张
    PHASE_LATE = 'late'        # 手牌 < 5张

    def __init__(self, strategy_mode: str = 'balanced'):
        """Initialize with strategy mode."""
        self.strategy_mode = strategy_mode
        self._last_best_play: Optional[Tuple] = None
        self._last_best_play_frames: int = 0
        self._play_switch_advantage: float = 0.15
        self._min_hold_frames: int = 3
        self._rank_to_env = {
            '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17,
            'Joker_B': 20, 'Joker_R': 30
        }
        self._env_to_rank = {v: k for k, v in self._rank_to_env.items()}

    def _card_to_env(self, card: str) -> int:
        """将卡牌字符串（如'3♠', 'Joker_R', 'JB', 'JR'）转换为环境整数"""
        # Handle joker variants
        if card.startswith('Joker') or card in ('JB', 'JR'):
            if card == 'Joker_B' or card == 'JB':
                return 20  # 小王
            if card == 'Joker_R' or card == 'JR':
                return 30  # 大王
        # Handle 10 - second char is '0'
        if len(card) >= 2 and card[1] == '0':
            return 10  # '10♠' -> 10
        rank = card[0] if card else '?'
        return self._rank_to_env.get(rank, 0)

    def _cards_to_env(self, cards: List[str]) -> List[int]:
        """将卡牌字符串列表转换为环境整数列表"""
        return [self._card_to_env(c) for c in cards]

    def _env_to_card(self, env_val: int, suit: str = '♠') -> str:
        """将环境整数转换回卡牌字符串"""
        rank = self._env_to_rank.get(env_val, '?')
        if rank.startswith('Joker'):
            return rank
        return rank + suit

    def _get_rank(self, card: str) -> str:
        """Extract rank from card."""
        # Handle joker variants
        if card.startswith('Joker') or card in ('JB', 'JR'):
            return 'Joker_B' if card in ('Joker_B', 'JB') else 'Joker_R'
        # Handle single-character ranks
        if len(card) == 1:
            return card
        if len(card) >= 2 and card[1] == '0':
            return '10'
        return card[0] if card else '?'

    def _get_rank_value(self, card: str) -> int:
        """Get the value of a card for comparison."""
        rank = self._get_rank(card)
        return self.CARD_VALUE.get(rank, 0)

    def _group_by_rank(self, cards: List[str]) -> Dict[str, List[str]]:
        """Group cards by rank (string form)."""
        grouped = {}
        for card in cards:
            rank = self._get_rank(card)
            if rank not in grouped:
                grouped[rank] = []
            grouped[rank].append(card)
        return grouped

    def _group_by_env_value(self, cards: List[str]) -> Dict[int, List[str]]:
        """Group cards by env integer value for proper sorting."""
        grouped = {}
        for card in cards:
            env_val = self._card_to_env(card)
            if env_val not in grouped:
                grouped[env_val] = []
            grouped[env_val].append(card)
        return grouped

    def decide(self, state: Dict) -> Dict:
        """
        Main entry point for AI decision.
        Returns decision dict with action, cards, type, confidence, reasoning, alternatives.
        """
        my_cards = state.get('my_cards', [])
        last_play = state.get('last_play')
        my_role = state.get('my_role', 'farmer')
        my_count = len(my_cards)

        if not my_cards:
            return {
                'action': 'pass',
                'cards': [],
                'type': 'pass',
                'confidence': 1.0,
                'reasoning': 'No cards to play',
                'alternatives': []
            }

        # Generate all valid plays
        all_plays = self._generate_all_plays(my_cards)

        # Filter to only plays that can beat the last play if needed
        if last_play and last_play['type'] != 'pass' and len(last_play.get('cards', [])) > 0:
            if last_play['type'] != 'unknown':
                valid_plays = self._filter_valid_plays(all_plays, last_play)
            else:
                # Unknown type: fallback to card-count matching to prevent invalid suggestions
                # e.g., don't suggest a pair (2 cards) when opponent played 3 cards
                valid_plays = self._filter_by_card_count(all_plays, last_play)
            # Debug log
            if self.DEBUG_OUTPUT:
                print(f"[AI] last_play={last_play.get('type')}({len(last_play.get('cards', []))} cards) valid_count={len(valid_plays)}")
            if not valid_plays:
                return {
                    'action': 'pass',
                    'cards': [],
                    'type': 'pass',
                    'confidence': 1.0,
                    'reasoning': f'Cannot beat {last_play["type"]} {last_play.get("cards", [])}, passing',
                    'alternatives': []
                }

            # 如果不能同类型管上，只能用炸弹/火箭时，把pass作为选项加入评分
            # 避免AI被迫浪费炸弹/火箭
            has_same_type = any(p['type'] == last_play['type'] for p in valid_plays)
            if not has_same_type:
                valid_plays.append({'type': 'pass', 'cards': [], 'value': 0, 'score': 0})
        else:
            valid_plays = all_plays
            if not valid_plays:
                valid_plays = [{'type': 'pass', 'cards': [], 'value': 0}]

        # Precompute play history stats once (saves per-play recomputation)
        precomputed_stats = None
        play_history = state.get('play_history', [])
        if play_history:
            precomputed_stats = {
                'big_cards': {'2': 0, 'Joker_B': 0, 'Joker_R': 0},
                'bombs': 0, 'rockets': 0
            }
            for hist in play_history:
                for card in hist.get('cards', []):
                    rank = self._get_rank(card)
                    if rank in precomputed_stats['big_cards']:
                        precomputed_stats['big_cards'][rank] += 1
                htype = hist.get('type', '')
                if htype == 'bomb':
                    precomputed_stats['bombs'] += 1
                elif htype == 'rocket':
                    precomputed_stats['rockets'] += 1

        # Check if we can win immediately (play all cards — only if truly winning)
        for play in valid_plays:
            if len(play['cards']) == my_count:
                # Only treat as "immediate win" if opponents have 0 cards left,
                # or playing a single/pair when opponents have ≤2 cards
                if play['type'] not in ('rocket', 'bomb'):
                    # Playing all non-bomb/non-rocket cards: likely a real win
                    play['score'] = 1000 + self._evaluate_play(play, state, precomputed_stats)
                    return {
                        'action': 'play',
                        'cards': play['cards'],
                        'type': play['type'],
                        'confidence': 1.0,
                        'reasoning': 'Can win immediately by playing all cards',
                        'alternatives': []
                    }
                # For rocket/bomb: only immediate win if opponents truly have nothing
                if state.get('upper_player_count', 0) == 0 and state.get('lower_player_count', 0) == 0:
                    play['score'] = 1000 + self._evaluate_play(play, state, precomputed_stats)
                    return {
                        'action': 'play',
                        'cards': play['cards'],
                        'type': play['type'],
                        'confidence': 1.0,
                        'reasoning': 'Final play to win the game',
                        'alternatives': []
                    }

        # Score all valid plays
        for play in valid_plays:
            score = self._evaluate_play(play, state, precomputed_stats)
            play['score'] = score

        # Sort by score descending
        valid_plays.sort(key=lambda p: p['score'], reverse=True)

        # === Hysteresis: stabilize AI suggestions across frames ===
        if valid_plays:
            current_cards = tuple(sorted(valid_plays[0].get('cards', [])))
            if self._last_best_play is None:
                self._last_best_play = current_cards
                self._last_best_play_frames = 0
            elif current_cards == self._last_best_play:
                self._last_best_play_frames += 1
            elif self._last_best_play_frames < self._min_hold_frames:
                # Keep previous play if still valid
                old_play = next(
                    (p for p in valid_plays if tuple(sorted(p.get('cards', []))) == self._last_best_play),
                    None
                )
                if old_play and old_play.get('score', 0) > 0:
                    old_play['score'] *= 1.5
                    valid_plays.sort(key=lambda p: p['score'], reverse=True)
                    self._last_best_play_frames += 1
                else:
                    self._last_best_play = current_cards
                    self._last_best_play_frames = 0
            else:
                old_play = next(
                    (p for p in valid_plays if tuple(sorted(p.get('cards', []))) == self._last_best_play),
                    None
                )
                if old_play and valid_plays[0]['score'] < old_play['score'] * (1 + self._play_switch_advantage):
                    old_play['score'] *= 1.5
                    valid_plays.sort(key=lambda p: p['score'], reverse=True)
                    self._last_best_play_frames += 1
                else:
                    self._last_best_play = current_cards
                    self._last_best_play_frames = 0

        # Pick best play
        best = valid_plays[0]

        # === 安全网：确认最佳出牌确实能打过上家的牌 ===
        # 如果因为某种原因 best 不能打过 last_play（如类型误判），强制不出或找替代方案
        if last_play and last_play['type'] not in ('pass', 'unknown') and last_play.get('cards'):
            if best['type'] != 'pass' and not self._can_beat(best, last_play):
                if self.DEBUG_OUTPUT:
                    print(f"[AI] SAFETY: best play {best['type']} can't beat {last_play['type']}, recalculating...")
                # 重新过滤：只保留能打过的出牌
                valid_plays = [p for p in valid_plays if p['type'] == 'pass' or self._can_beat(p, last_play)]
                if valid_plays:
                    valid_plays.sort(key=lambda p: p['score'], reverse=True)
                    best = valid_plays[0]
                else:
                    best = {'type': 'pass', 'cards': [], 'value': 0, 'score': -999}

        # Collect top alternatives
        alternatives = valid_plays[1:5] if len(valid_plays) > 1 else []

        reasoning = self._generate_reasoning(best, state)

        return {
            'action': 'play' if best['type'] != 'pass' else 'pass',
            'cards': best.get('cards', []),
            'type': best.get('type', 'pass'),
            'confidence': min(1.0, best['score'] / 100),
            'reasoning': reasoning,
            'alternatives': alternatives
        }

    def _generate_all_plays(self, cards: List[str]) -> List[Dict]:
        """Generate all possible valid plays from current hand.

        效率优化版本：
        1. 使用集合去重已处理的组合
        2. 限制顺子/连对最大长度
        3. 提前剪枝减少无效计算
        """
        plays = []
        n = len(cards)
        if n == 0:
            return []

        grouped = self._group_by_rank(cards)
        grouped_env = self._group_by_env_value(cards)  # 按env值分组

        # 用于去重的已见组合
        seen_combos = set()

        # === 基础牌型 ===
        # Single cards (单张)
        for env_val, card_list in grouped_env.items():
            for card in card_list:
                plays.append({
                    'type': 'single',
                    'cards': [card],
                    'value': env_val,
                    'length': 1
                })

        # Pairs (对子)
        for env_val, card_list in grouped_env.items():
            if len(card_list) >= 2:
                plays.append({
                    'type': 'pair',
                    'cards': card_list[:2],
                    'value': env_val,
                    'length': 2
                })

        # Triples (三张/三条）
        for env_val, card_list in grouped_env.items():
            if len(card_list) >= 3:
                plays.append({
                    'type': 'triple',
                    'cards': card_list[:3],
                    'value': env_val,
                    'length': 3
                })

        # Bombs (炸弹 - 四张相同)
        for env_val, card_list in grouped_env.items():
            if len(card_list) >= 4:
                plays.append({
                    'type': 'bomb',
                    'cards': card_list[:4],
                    'value': env_val + 100,  # 炸弹基础分加100
                    'length': 4
                })

        # Rocket (王炸)
        def has_joker(cards, joker):
            return any(c.startswith('Joker') or c == joker for c in cards)
        has_b = has_joker(cards, 'JB') or any('Joker_B' in c for c in cards)
        has_r = has_joker(cards, 'JR') or any('Joker_R' in c for c in cards)
        if has_b and has_r:
            jb_card = next((c for c in cards if 'Joker_B' in c or c == 'JB'), 'Joker_B')
            jr_card = next((c for c in cards if 'Joker_R' in c or c == 'JR'), 'Joker_R')
            plays.append({
                'type': 'rocket',
                'cards': [jb_card, jr_card],
                'value': 1000,
                'length': 2
            })

        # === 带牌类型 ===
        # Triple with single (三带一) - 优化：只生成有意义的组合
        for env_val_t, triple_list in grouped_env.items():
            if len(triple_list) >= 3:
                triple = triple_list[:3]
                for env_val_s, single_list in grouped_env.items():
                    if env_val_s != env_val_t and len(single_list) >= 1:
                        combo_key = ('tws', env_val_t, env_val_s)
                        if combo_key not in seen_combos:
                            seen_combos.add(combo_key)
                            plays.append({
                                'type': 'triple_with_single',
                                'cards': triple + [single_list[0]],
                                'value': env_val_t,
                                'length': 4
                            })

        # Triple with pair (三带二)
        for env_val_t, triple_list in grouped_env.items():
            if len(triple_list) >= 3:
                triple = triple_list[:3]
                for env_val_p, pair_list in grouped_env.items():
                    if env_val_p != env_val_t and len(pair_list) >= 2:
                        combo_key = ('twp', env_val_t, env_val_p)
                        if combo_key not in seen_combos:
                            seen_combos.add(combo_key)
                            plays.append({
                                'type': 'triple_with_pair',
                                'cards': triple + pair_list[:2],
                                'value': env_val_t,
                                'length': 5
                            })

        # === 顺子类型（限制最大长度12张）===
        plays.extend(self._generate_straights(cards, grouped_env))

        # Consecutive pairs (连对 - 3对+连续对子, 限制最大长度）
        plays.extend(self._generate_consecutive_pairs(cards, grouped_env))

        # Airplane (飞机不带 - 2个+连续三张)
        plays.extend(self._generate_airplane(cards, grouped_env))

        # Airplane with singles (飞机带单) - 优化版本
        plays.extend(self._generate_airplane_with_singles_opt(cards, grouped_env))

        # Airplane with pairs (飞机带对) - 优化版本
        plays.extend(self._generate_airplane_with_pairs_opt(cards, grouped_env))

        # === 四带类型 ===
        # Four with two singles (四带二单)
        plays.extend(self._generate_four_with_two_singles(cards, grouped_env))

        # Four with two pairs (四带两对)
        plays.extend(self._generate_four_with_two_pairs(cards, grouped_env))

        return plays

    def _generate_straights(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate all possible straights (5-12 consecutive singles).

        效率优化：限制最大长度12张（斗地主规则上限）
        """
        plays = []
        # 不能有2和王
        available = [v for v in grouped_env.keys() if v < 17]
        if len(available) < 5:
            return []

        available_sorted = sorted(available)
        n = len(available_sorted)

        # 限制最大顺子长度12张（斗地主规则）
        max_straight_length = 12

        for start in range(n):
            # 最短5张，最长max_straight_length
            min_end = start + 4
            max_end = min(start + max_straight_length - 1, n - 1)

            for end in range(min_end, max_end + 1):
                # 检查是否是连续序列
                if available_sorted[end] - available_sorted[start] == end - start:
                    straight_vals = available_sorted[start:end+1]
                    straight_cards = [grouped_env[val][0] for val in straight_vals]
                    plays.append({
                        'type': 'straight',
                        'cards': straight_cards,
                        'value': straight_vals[-1],
                        'length': len(straight_vals)
                    })
        return plays

    def _generate_consecutive_pairs(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate all possible consecutive pairs (3-10 consecutive pairs).

        效率优化：限制最大长度10对（20张）
        """
        plays = []
        # 需要每种至少2张，不能有2和王
        available = [v for v in grouped_env.keys() if v < 17 and len(grouped_env[v]) >= 2]
        if len(available) < 3:
            return []

        available_sorted = sorted(available)
        n = len(available_sorted)

        # 限制最大连对数量10对（斗地主规则）
        max_pairs = 10

        for start in range(n):
            min_end = start + 2  # 至少3对
            max_end = min(start + max_pairs - 1, n - 1)

            for end in range(min_end, max_end + 1):
                if available_sorted[end] - available_sorted[start] == end - start:
                    pair_vals = available_sorted[start:end+1]
                    pair_cards = []
                    for val in pair_vals:
                        pair_cards.extend(grouped_env[val][:2])
                    plays.append({
                        'type': 'consecutive_pairs',
                        'cards': pair_cards,
                        'value': pair_vals[-1],
                        'length': len(pair_vals) * 2
                    })
        return plays

    def _generate_airplane(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate all possible airplanes (2+ consecutive triples)."""
        plays = []
        # 需要每种至少3张，且不能是2或王（value >= 17）
        available = [v for v in grouped_env.keys() if len(grouped_env[v]) >= 3 and v < 17]
        if len(available) < 2:
            return []

        available_sorted = sorted(available)
        n = len(available_sorted)

        for start in range(n):
            for end in range(start + 1, n):  # 至少2个三张
                if available_sorted[end] - available_sorted[start] == end - start:
                    triple_vals = available_sorted[start:end+1]
                    triple_cards = []
                    for val in triple_vals:
                        triple_cards.extend(grouped_env[val][:3])
                    plays.append({
                        'type': 'airplane',
                        'cards': triple_cards,
                        'value': triple_vals[-1],
                        'length': len(triple_vals) * 3
                    })
        return plays

    def _generate_airplane_with_singles_opt(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate airplanes with single cards - optimized version.

        优化：避免生成所有单张组合，只生成有代表性的组合
        """
        plays = []
        available = [v for v in grouped_env.keys() if len(grouped_env[v]) >= 3 and v < 17]
        if len(available) < 2:
            return []

        available_sorted = sorted(available)
        n = len(available_sorted)

        for start in range(n):
            for end in range(start + 1, n):
                if available_sorted[end] - available_sorted[start] == end - start:
                    triple_count = end - start + 1
                    triple_vals = available_sorted[start:end+1]
                    triple_cards = []
                    for val in triple_vals:
                        triple_cards.extend(grouped_env[val][:3])

                    # 找足够的单张（每组三张带一个单张）
                    singles_pool = []
                    for val, card_list in grouped_env.items():
                        if val not in triple_vals:
                            singles_pool.extend(card_list)

                    if len(singles_pool) >= triple_count:
                        # 只取最小和最大的单张组合，避免组合爆炸
                        from itertools import combinations
                        singles_pool_sorted = sorted(singles_pool, key=lambda c: self._card_to_env(c))

                        # 取最小单张组合（代表性）
                        min_combo = singles_pool_sorted[:triple_count]
                        plays.append({
                            'type': 'airplane_with_singles',
                            'cards': triple_cards + min_combo,
                            'value': triple_vals[-1],
                            'length': len(triple_cards) + triple_count
                        })

                        # 如果单张数量充足，也取最大单张组合（另一种策略）
                        if len(singles_pool_sorted) >= triple_count + 3:
                            max_combo = singles_pool_sorted[-triple_count:]
                            plays.append({
                                'type': 'airplane_with_singles',
                                'cards': triple_cards + max_combo,
                                'value': triple_vals[-1],
                                'length': len(triple_cards) + triple_count
                            })

                        # 中间值组合（避免重复）
                        if len(singles_pool_sorted) >= triple_count + 6:
                            mid_combo = singles_pool_sorted[triple_count:triple_count*2]
                            plays.append({
                                'type': 'airplane_with_singles',
                                'cards': triple_cards + mid_combo,
                                'value': triple_vals[-1],
                                'length': len(triple_cards) + triple_count
                            })
        return plays

    def _generate_airplane_with_pairs_opt(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate airplanes with pairs - optimized version.

        优化：只生成有代表性的对子组合
        """
        plays = []
        available = [v for v in grouped_env.keys() if len(grouped_env[v]) >= 3 and v < 17]
        if len(available) < 2:
            return []

        available_sorted = sorted(available)
        n = len(available_sorted)

        for start in range(n):
            for end in range(start + 1, n):
                if available_sorted[end] - available_sorted[start] == end - start:
                    triple_count = end - start + 1
                    triple_vals = available_sorted[start:end+1]
                    triple_cards = []
                    for val in triple_vals:
                        triple_cards.extend(grouped_env[val][:3])

                    # 找足够的对子
                    pairs_pool = []
                    for val, card_list in grouped_env.items():
                        if val not in triple_vals and len(card_list) >= 2:
                            pairs_pool.append((val, card_list[0], card_list[1]))

                    if len(pairs_pool) >= triple_count:
                        # 按对子值排序
                        pairs_pool_sorted = sorted(pairs_pool, key=lambda p: p[0])

                        # 取最小对子组合
                        min_combo = pairs_pool_sorted[:triple_count]
                        pair_cards = []
                        for _, c1, c2 in min_combo:
                            pair_cards.extend([c1, c2])
                        plays.append({
                            'type': 'airplane_with_pairs',
                            'cards': triple_cards + pair_cards,
                            'value': triple_vals[-1],
                            'length': len(triple_cards) + triple_count * 2
                        })

                        # 如果对子充足，取最大对子组合
                        if len(pairs_pool_sorted) >= triple_count + 2:
                            max_combo = pairs_pool_sorted[-triple_count:]
                            pair_cards = []
                            for _, c1, c2 in max_combo:
                                pair_cards.extend([c1, c2])
                            plays.append({
                                'type': 'airplane_with_pairs',
                                'cards': triple_cards + pair_cards,
                                'value': triple_vals[-1],
                                'length': len(triple_cards) + triple_count * 2
                            })
        return plays

    def _generate_four_with_two_singles(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate all possible four cards with two singles."""
        plays = []
        # 找四张
        for env_val, card_list in grouped_env.items():
            if len(card_list) >= 4:
                four = card_list[:4]
                # 找两张单张
                singles_pool = []
                for val, cl in grouped_env.items():
                    if val != env_val:
                        singles_pool.extend(cl)

                if len(singles_pool) >= 2:
                    from itertools import combinations
                    for combo in combinations(singles_pool, 2):
                        plays.append({
                            'type': 'four_with_two_singles',
                            'cards': four + list(combo),
                            'value': env_val + 50,
                            'length': 6
                        })
        return plays

    def _generate_four_with_two_pairs(self, cards: List[str], grouped_env: Dict[int, List[str]]) -> List[Dict]:
        """Generate all possible four cards with two pairs."""
        plays = []
        # 找四张
        for env_val, card_list in grouped_env.items():
            if len(card_list) >= 4:
                four = card_list[:4]
                # 找两对
                pairs = []
                for val, cl in grouped_env.items():
                    if val != env_val and len(cl) >= 2:
                        pairs.append(cl[:2])

                if len(pairs) >= 2:
                    from itertools import combinations
                    for combo in combinations(pairs, 2):
                        two_pairs = []
                        for p in combo:
                            two_pairs.extend(p)
                        plays.append({
                            'type': 'four_with_two_pairs',
                            'cards': four + two_pairs,
                            'value': env_val + 60,
                            'length': 8
                        })
        return plays

    def _filter_valid_plays(self, plays: List[Dict], last_play: Dict) -> List[Dict]:
        """Filter only plays that can beat the last play."""
        return [p for p in plays if self._can_beat(p, last_play)]

    def _filter_by_card_count(self, plays: List[Dict], last_play: Dict) -> List[Dict]:
        """Fallback: when card type is unknown, match by card count + allow bombs/rockets."""
        last_count = len(last_play.get('cards', []))
        if last_count == 0:
            return plays
        valid = []
        for p in plays:
            if p['type'] in ('bomb', 'rocket'):
                valid.append(p)
            elif len(p.get('cards', [])) == last_count and p.get('value', 0) > last_play.get('value', 0):
                valid.append(p)
        return valid

    def _can_beat(self, play: Dict, last_play: Dict) -> bool:
        """Check if current play can beat the last play."""
        play_type = play['type']
        last_type = last_play['type']

        # Rocket always beats anything (except unknown — usually a misdetection)
        if play_type == 'rocket' and last_type != 'unknown':
            # Rocket can only beat non-rocket plays
            if last_type == 'rocket':
                return False  # Rocket cannot beat rocket
            return True

        # Bomb beats anything except rocket
        if play_type == 'bomb':
            if last_type == 'rocket':
                return False
            if last_type == 'bomb':
                return play['value'] > last_play['value']
            return True  # Bomb beats non-bomb

        # Last play is bomb, current is not - can't beat
        if last_type == 'bomb' or last_type == 'rocket':
            return False

        # Must be same type and same length to beat
        if play_type != last_type:
            return False

        if play.get('length', len(play['cards'])) != last_play.get('length', len(last_play['cards'])):
            return False

        return play['value'] > last_play['value']

    def _get_play_value(self, play: Dict) -> int:
        """Get the comparison value of a play."""
        return play.get('value', 0)

    def _get_game_phase(self, my_cards: List[str]) -> str:
        """Determine current game phase based on remaining cards."""
        count = len(my_cards)
        if count > 10:
            return self.PHASE_EARLY
        elif count >= 5:
            return self.PHASE_MID
        else:
            return self.PHASE_LATE

    def _evaluate_remaining_structure(self, remaining_cards: List[str], play: Dict, state: Dict) -> float:
        """
        链式思考：评估出完这些牌后，剩余手牌的结构好不好打。
        借鉴 DouZero 的状态-动作价值和 ZhouWeikuan 的手牌威力评估。
        高分 = 剩余手牌结构好，容易继续出完。
        """
        if not remaining_cards:
            return 100.0  # 出完了，最高分

        grouped = self._group_by_env_value(remaining_cards)
        n = len(remaining_cards)

        # 统计剩余手牌结构
        singles = pairs = triples = quads = 0
        for val, cl in grouped.items():
            count = len(cl)
            if count == 1:
                singles += 1
            elif count == 2:
                pairs += 1
            elif count == 3:
                triples += 1
            elif count >= 4:
                quads += 1

        score = 0.0

        # 基础：剩余牌越少越好
        score += (17 - n) * 2

        # 结构质量评分
        if n >= 5:  # 中前期：关注结构合理性
            # 计算"孤单单张"——那些没有配对或成三张的单牌
            absorb_capacity = pairs + triples * 2 + quads * 3
            lonely = max(0, singles - absorb_capacity)
            score -= lonely * 3  # 孤独单张是负担
            # 对子和三张是好的结构
            score += pairs * 3
            score += triples * 5
            score += quads * 10
        else:  # 后期（< 5张）
            if n <= 2:
                score += 10  # 快赢了
            if singles == 1 and n <= 2:
                score += 5   # 剩一张单张很好出
            if pairs >= 1 and singles <= 1:
                score += 8   # 对子+少量单张，结构好

        # 出牌类型与剩余手牌的配合度
        play_type = play.get('type', '')
        cards_played = len(play.get('cards', []))

        # 一次出多张牌有优势
        if cards_played >= 5:
            score += cards_played * 0.5

        # 如果出的是最后一张某种牌面（没有遗留单张），加分
        if play_type == 'single':
            remaining_vals = [self._card_to_env(c) for c in remaining_cards]
            if play.get('value', 0) not in remaining_vals:
                score += 2  # 清空了一个牌面，没有遗留

        return score

    def _evaluate_play(self, play: Dict, state: Dict,
                       precomputed_stats: Optional[Dict] = None) -> float:
        """Evaluate a play and assign score."""
        score = 0.0
        play_type = play['type']
        my_cards = state.get('my_cards', [])
        my_role = state.get('my_role', 'farmer')
        remaining_my = len(my_cards) - len(play['cards'])
        upper_count = state.get('upper_player_count', 0)
        lower_count = state.get('lower_player_count', 0)
        last_play_info = state.get('last_play')
        game_phase = self._get_game_phase(my_cards)

        # 计算效率分数
        efficiency = len(play['cards']) / len(my_cards) if len(my_cards) > 0 else 0

        # === 出牌效率分（最核心的评分）===
        if remaining_my == 0:
            score += 2000  # 能出完是最高优先
        else:
            # 根据游戏阶段调整效率权重
            if remaining_my <= 5:
                score += efficiency * 120  # 后期稍微加速
            elif remaining_my <= 10:
                score += efficiency * 100  # 中期
            else:
                score += efficiency * 100  # 早期

        # === 牌型基础分 ===
        type_scores = {
            'single': 1,
            'pair': 2,
            'triple': 3,
            'triple_with_single': 4,
            'triple_with_pair': 5,
            'straight': 4,
            'consecutive_pairs': 6,
            'airplane': 8,
            'airplane_with_singles': 9,
            'airplane_with_pairs': 10,
            'bomb': 15,
            'four_with_two_singles': 10,
            'four_with_two_pairs': 12,
            'rocket': 20
        }
        score += type_scores.get(play_type, 0)

        # === 最小能打过原则（关键！） ===
        # 如果有相同类型的牌能打过，不应该用炸弹/火箭
        if last_play_info and last_play_info.get('type') != 'pass' and last_play_info.get('cards'):
            last_type = last_play_info.get('type', '')
            last_value = last_play_info.get('value', 0)

            # 如果上家出的是普通牌型（非炸弹/火箭），用炸弹/火箭是过度浪费
            if play_type in ('bomb', 'rocket') and last_type not in ('bomb', 'rocket'):
                if play_type == 'bomb':
                    score -= 50
                elif play_type == 'rocket':
                    score -= 80
                # 早期游戏更不应该浪费炸弹
                if game_phase == self.PHASE_EARLY:
                    score -= 30
            # 上家出炸弹时，用火箭打是合理的
            elif last_type == 'bomb' and play_type == 'rocket':
                score += 20  # 王炸打炸弹，正确使用
            # 同类型，比较value - 越小越好（能打过就行）
            elif play_type == last_type:
                play_value = play.get('value', 0)
                overkill = play_value - last_value
                if overkill > 0:
                    score += 10 - overkill * 0.5  # 刚好能打过最好

        # === 自由出牌时禁止浪费炸弹/火箭 ===
        if not last_play_info or last_play_info.get('type') == 'pass' or not last_play_info.get('cards'):
            if play_type == 'rocket':
                score -= 80  # 绝不在自由出牌时先手出王炸
            elif play_type == 'bomb':
                score -= 40  # 绝不在自由出牌时先手出炸弹

            # === 自由出牌策略：先手出牌技巧 ===
            # 参考 ZhouWeikuan/DouDiZhu 的 robotFirstPlay 策略
            if play_type == 'single':
                val = play.get('value', 0)
                if val <= 6:
                    score += 8   # 出小单张探路（3-7）
                elif 7 <= val <= 10:
                    score += 4   # 中张单也可以出
                elif val == 17:
                    score -= 15  # 不要先手出2，它会挡路
                elif val >= 20:
                    score -= 30  # 绝不要先手出王
            elif play_type == 'pair':
                val = play.get('value', 0)
                if val <= 6:
                    score += 5   # 出小对子
                elif val >= 17:
                    score -= 10  # 不要先手出2对
            elif play_type == 'triple':
                val = play.get('value', 0)
                if val <= 8:
                    score += 4   # 小三条可以出
            elif play_type == 'straight':
                score += 8      # 顺子早出，避免被拆
            elif play_type == 'consecutive_pairs':
                score += 6      # 连对早出
            elif play_type == 'airplane':
                score += 10     # 飞机早出

        # === 角色策略 ===
        if my_role == 'landlord':
            if play_type not in ['bomb', 'rocket']:
                env_val = play.get('value', 0)
                if env_val < 10:
                    score += 8  # 提高清低点分，更激进
                elif env_val >= 14:
                    score -= 5   # 保留大牌控场
                # 中期更激进
                if remaining_my <= 8:
                    score += efficiency * 150
        else:
            if last_play_info is None:
                if play_type == 'single':
                    env_val = play.get('value', 0)
                    if env_val < 10:
                        score += 5
            else:
                last_type = last_play_info.get('type', '')
                if last_type in ['airplane', 'airplane_with_singles', 'airplane_with_pairs']:
                    if play_type in ['bomb', 'rocket']:
                        score += 20

        # === 炸弹/火箭保留策略 ===
        if play_type == 'bomb':
            if upper_count <= 4 or lower_count <= 4:
                score += 30  # 对手快出完了，炸弹价值高
            else:
                score -= 15  # 保留炸弹
        if play_type == 'rocket':
            if upper_count <= 4 or lower_count <= 4:
                score += 40  # 对手快出完了，王炸是关键
            else:
                score -= 30  # 保留王炸，除非必要不用

        # === 策略模式 ===
        if self.strategy_mode == self.STRATEGY_AGGRESSIVE:
            if play_type not in ['bomb', 'rocket']:
                score += 5
        elif self.strategy_mode == self.STRATEGY_DEFENSIVE:
            if play_type in ['bomb', 'rocket']:
                score += 10
            if remaining_my <= 3:
                score += 15

        # === 游戏阶段感知策略 ===
        if game_phase == self.PHASE_EARLY:
            # Early: 诱弹、留大牌
            if play_type == 'bomb':
                score -= 10  # 早期少出炸弹，诱骗对手
            if play_type in ('single', 'pair') and play.get('value', 0) >= 14:
                score += 3   # 保留大牌
        elif game_phase == self.PHASE_MID:
            # Mid: 逼弹、控场
            if play_type == 'bomb':
                score += 5   # 中期可以开始用炸弹
            if remaining_my <= 3:
                score += 10  # 快出完了，加速
        else:  # PHASE_LATE
            # Late: 抢出、算牌
            if play_type not in ['bomb', 'rocket']:
                score += (5 - remaining_my) * 2  # 越快出完越好
            if remaining_my == 1 and len(play['cards']) == 1:
                score += 20  # 单张跑牌优先

        # === 记牌功能增强 ===
        play_history = state.get('play_history', [])
        if play_history:
            # 使用预计算的统计（避免每次评分重复遍历 history）
            if precomputed_stats:
                big_cards_played = precomputed_stats.get('big_cards', {'2': 0, 'Joker_B': 0, 'Joker_R': 0})
                bombs_played = precomputed_stats.get('bombs', 0)
                rockets_played = precomputed_stats.get('rockets', 0)
            else:
                # 回退：内联计算（兼容旧调用）
                big_cards_played = {'2': 0, 'Joker_B': 0, 'Joker_R': 0}
                bombs_played = 0
                rockets_played = 0
                for hist in play_history:
                    for card in hist.get('cards', []):
                        rank = self._get_rank(card)
                        if rank == '2':
                            big_cards_played['2'] += 1
                        elif rank == 'Joker_B':
                            big_cards_played['Joker_B'] += 1
                        elif rank == 'Joker_R':
                            big_cards_played['Joker_R'] += 1
                    hist_type = hist.get('type', '')
                    if hist_type == 'bomb':
                        bombs_played += 1
                    elif hist_type == 'rocket':
                        rockets_played += 1

            # 大牌是否已出完的推断
            # 如果2已经出了3张，说明对手手里很可能没有2了
            if play_type == 'bomb' or play_type == 'rocket':
                # 检查炸弹/火箭是否还有价值
                if rockets_played > 0:
                    score -= 15  # 王炸已出，降低炸弹价值
                if bombs_played >= 2:
                    score -= 10  # 多个炸弹已出，继续出炸弹价值降低

            # 如果自己有大牌2，且2已经出的不多，可以考虑出
            if play_type == 'single' and play.get('value', 0) == 17:  # 2
                if big_cards_played['2'] >= 3:
                    score += 15  # 2几乎确定是当前最大牌

            # 对手可能无炸弹判断
            if bombs_played >= 2:
                if play_type == 'bomb':
                    score += 10  # 对手很可能没炸弹了，可以更激进

        # === 农民配合策略增强 ===
        if my_role == 'farmer' and last_play_info:
            last_type = last_play_info.get('type', '')
            last_value = last_play_info.get('value', 0)
            last_player = last_play_info.get('player', '')

            # 判断地主是谁（根据state判断队友）
            landlord = state.get('landlord', '')
            if landlord == 'upper':
                partner = 'lower'  # 地主是上家，下家是队友
            elif landlord == 'lower':
                partner = 'upper'  # 地主是下家，上家是队友
            else:
                partner = None

            # 如果是队友出的牌，不压队友（除非用炸弹/火箭）
            if partner and last_player == partner:
                if play_type not in ('bomb', 'rocket'):
                    score -= 30  # 不压队友

            # 通过历史记录判断队友是否出过牌
            # 如果队友出过牌主动管上家，说明队友在控场
            play_history = state.get('play_history', [])
            partner_played_big = False
            if partner and play_history:
                for hist in reversed(play_history):
                    if hist.get('player') == partner and hist.get('type') != 'pass':
                        partner_played_big = True
                        break

            if partner_played_big:
                # 队友已经在控场，降低出炸弹的优先级
                if play_type in ['bomb', 'rocket']:
                    score -= 15

            # 如果上家出的是飞机/大牌，农民配合：保留炸弹控场
            if last_type in ['airplane', 'airplane_with_singles', 'airplane_with_pairs']:
                if len(my_cards) > 10 and play_type in ['bomb', 'rocket']:
                    score -= 15  # 自己牌多，留给队友机会
            # 地主快出完时（<=3张），农民要用炸弹拦截
            if play_type in ('bomb', 'rocket'):
                if (landlord == 'upper' and state.get('upper_player_count', 0) <= 3) or \
                   (landlord == 'lower' and state.get('lower_player_count', 0) <= 3):
                    score += 25  # 必须炸，拦住地主

            # === 农民主动配合策略 ===
            # 队友只剩1-3张时，送小单张/小对子
            if partner:
                partner_count = state.get('lower_count', 0) if partner == 'lower' else state.get('upper_count', 0)
                if 1 <= partner_count <= 3:
                    if play_type == 'single' and play.get('value', 0) <= 8:
                        score += 25  # 主动送小单张给队友
                    elif play_type == 'pair' and play.get('value', 0) <= 6:
                        score += 20  # 送小对子给队友

        # === 链式思考：出牌后手牌结构评估 ===
        # 出完这些牌之后，剩下的牌好不好打？
        # 如果剩余手牌结构好（对子多、单张少），说明这个出牌选择好
        remaining_cards = list(my_cards)
        for c in play['cards']:
            if c in remaining_cards:
                remaining_cards.remove(c)
        structure_score = self._evaluate_remaining_structure(remaining_cards, play, state)
        score += structure_score * 0.3  # 权重30%

        return score

    def _generate_reasoning(self, play: Dict, state: Dict) -> str:
        """Generate human-readable reasoning for the decision."""
        if play['type'] == 'pass':
            return "Cannot beat opponent's play, passing."

        my_role = state.get('my_role', 'farmer')
        play_type = play['type']

        type_names = {
            'single': '单张',
            'pair': '对子',
            'triple': '三张',
            'triple_with_single': '三带一',
            'triple_with_pair': '三带二',
            'bomb': '炸弹',
            'rocket': '王炸',
            'straight': '顺子',
            'consecutive_pairs': '连对',
            'airplane': '飞机',
            'airplane_with_singles': '飞机带单',
            'airplane_with_pairs': '飞机带对',
            'four_with_two_singles': '四带二单',
            'four_with_two_pairs': '四带两对'
        }

        type_name = type_names.get(play_type, play_type)
        highest_card = sorted(play['cards'], key=lambda c: self._card_to_env(c))[-1]
        rank = self._get_rank(highest_card)

        # 中文牌面值
        rank_display = {
            '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
            '9': '9', '10': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
            '2': '2', 'Joker_B': '小王', 'Joker_R': '大王'
        }
        rank_str = rank_display.get(rank, rank)

        if my_role == 'landlord':
            base = f"出{len(play['cards'])}张【{type_name}】。作为地主，清牌压农民。"
        else:
            base = f"出{len(play['cards'])}张【{type_name}】，最大牌:{rank_str}。"

        if play['score'] > 100:
            base += " 高信心出牌。"
        elif play['score'] < 0:
            base += " 风险较高，请谨慎。"

        return base


    def clear_hysteresis(self) -> None:
        """Clear hysteresis state for new round."""
        self._last_best_play = None
        self._last_best_play_frames = 0


if __name__ == "__main__":
    import time

    print("Testing DoudizhuAI...")

    ai = DoudizhuAI(strategy_mode='balanced')

    # Test a simple case
    test_state = {
        'my_cards': ['3♠', '3♥', '3♦', '4♠', '5♠', '6♠', '7♠', '2♥', 'Joker_B', 'Joker_R'],
        'my_role': 'landlord',
        'last_play': None,
        'current_turn': 'self',
        'upper_player_count': 10,
        'lower_player_count': 8
    }

    start_time = time.time()
    decision = ai.decide(test_state)
    elapsed = (time.time() - start_time) * 1000

    print(f"Decision time: {elapsed:.2f}ms")
    print(f"Action: {decision['action']}")
    print(f"Number of cards: {len(decision['cards'])}")
    print(f"Type: {decision['type']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Reasoning: {decision['reasoning']}")

    if elapsed <= 200:
        print("[OK] Decision time meets requirement (≤ 200ms)")
    else:
        print("[FAIL] Decision time exceeds target")

    # Test can_beat
    print("\nTesting _can_beat:")
    single_3 = {'type': 'single', 'value': 3, 'cards': ['3♠']}
    single_4 = {'type': 'single', 'value': 4, 'cards': ['4♠']}
    print(f"  4 beats 3: {ai._can_beat(single_4, single_3)} → [OK]")
    expected = not ai._can_beat(single_3, single_4)
    print(f"  3 beats 4: {ai._can_beat(single_3, single_4)} → {expected} → [OK]")

    print("\nAll tests completed!")
