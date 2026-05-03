"""
斗地主裁判模拟器
发牌给三个策略不同的AI玩家，协调出牌顺序，记录博弈过程，判定胜负。
"""

import random
import importlib.util
import os
from typing import List, Dict, Optional, Tuple

# 直接加载ai_engine避免modules/__init__.py触发cv2递归导入
_spec = importlib.util.spec_from_file_location('ai_engine', os.path.join('modules', 'ai_engine.py'))
_ai_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ai_mod)
DoudizhuAI = _ai_mod.DoudizhuAI

_spec2 = importlib.util.spec_from_file_location('state_manager', os.path.join('modules', 'state_manager.py'))
_sm_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_sm_mod)
GameStateManager = _sm_mod.GameStateManager

# 牌面值到显示名的映射
RANK_DISPLAY = {
    '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
    '9': '9', '10': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
    '2': '2', 'Joker_B': '小王', 'Joker_R': '大王'
}

# 完整牌组定义
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['S', 'H', 'D', 'C']
JOKERS = ['Joker_B', 'Joker_R']

def create_deck() -> List[str]:
    """创建54张牌的牌组"""
    deck = []
    for rank in RANKS:
        for suit in SUITS:
            deck.append(f"{rank}{suit}")
    deck.extend(JOKERS)
    return deck

def deal_cards(deck: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """发牌：返回三个玩家的手牌和底牌"""
    random.shuffle(deck)
    landlord_hand = deck[:17]
    farmer1_hand = deck[17:34]
    farmer2_hand = deck[34:51]
    bottom_cards = deck[51:]  # 3张底牌
    return landlord_hand, farmer1_hand, farmer2_hand, bottom_cards

def card_sort_key(card: str) -> int:
    """卡牌排序键值"""
    state = GameStateManager()
    return state.CARD_VALUE.get(state._get_rank(card), 0)

def format_cards(cards: List[str]) -> str:
    """格式化卡牌显示"""
    if not cards:
        return "空"
    sorted_cards = sorted(cards, key=card_sort_key)
    return ' '.join(sorted_cards)

class DoudizhuGame:
    """斗地主博弈模拟器"""

    def __init__(self):
        self.deck = create_deck()

        # 发牌结果
        self.landlord_hand = []
        self.farmer1_hand = []
        self.farmer2_hand = []
        self.bottom_cards = []

        # AI玩家（平衡=地主，激进=农民1，保守=农民2）
        self.landlord_ai = DoudizhuAI('balanced')
        self.farmer1_ai = DoudizhuAI('aggressive')
        self.farmer2_ai = DoudizhuAI('defensive')

        # 当前手牌状态
        self.hands = {
            'landlord': [],
            'farmer1': [],
            'farmer2': []
        }

        # 出牌状态
        self.current_player = 'landlord'  # 当前该谁出牌
        self.last_play = None  # 上一张非pass的出牌 {'player', 'cards', 'type', 'value'}
        self.last_play_player = None  # 最后出牌（非pass）的人
        self.pass_count = 0  # 连续pass的人数

        # 历史记录
        self.rounds = []  # 每轮的出牌记录
        self.current_round = []  # 当前轮的出牌

    def deal(self) -> Dict:
        """发牌并初始化游戏"""
        self.landlord_hand, self.farmer1_hand, self.farmer2_hand, self.bottom_cards = deal_cards(self.deck)

        # 排序后分配
        self.hands['landlord'] = sorted(self.landlord_hand, key=card_sort_key)
        self.hands['farmer1'] = sorted(self.farmer1_hand, key=card_sort_key)
        self.hands['farmer2'] = sorted(self.farmer2_hand, key=card_sort_key)

        self.current_player = 'landlord'
        self.last_play = None
        self.last_play_player = None
        self.pass_count = 0
        self.rounds = []
        self.current_round = []

        return {
            'landlord': self.hands['landlord'].copy(),
            'farmer1': self.hands['farmer1'].copy(),
            'farmer2': self.hands['farmer2'].copy(),
            'bottom': self.bottom_cards.copy()
        }

    def _get_ai_and_role(self, player: str) -> Tuple[DoudizhuAI, str]:
        """获取对应玩家的AI和角色"""
        if player == 'landlord':
            return self.landlord_ai, 'landlord'
        elif player == 'farmer1':
            return self.farmer1_ai, 'farmer'
        else:
            return self.farmer2_ai, 'farmer'

    def _get_partner(self, player: str) -> str:
        """获取队友"""
        if player == 'farmer1':
            return 'farmer2'
        elif player == 'farmer2':
            return 'farmer1'
        return None  # 地主没有队友

    def _get_opponent_landlord_position(self) -> str:
        """获取地主位置（在农民眼中）"""
        # 这个信息需要从游戏状态推断
        # 在这个模拟中，地主就是'landlord'
        return 'landlord'

    def _map_player_to_position(self, target: str, perspective: str) -> str:
        """将玩家名转换为从perspective视角的位置名 ('self'/'upper'/'lower')"""
        if target == perspective:
            return 'self'
        order = ['landlord', 'farmer1', 'farmer2']
        idx = order.index(perspective)
        if order[(idx + 1) % 3] == target:
            return 'lower'
        return 'upper'

    def _build_state(self, player: str) -> Dict:
        """为玩家构建游戏状态"""
        hand = self.hands[player]
        ai, my_role = self._get_ai_and_role(player)

        # 计算其他玩家剩余牌数
        if player == 'landlord':
            upper_count = len(self.hands['farmer1'])
            lower_count = len(self.hands['farmer2'])
            landlord_pos = 'self'
        elif player == 'farmer1':
            upper_count = len(self.hands['farmer2'])
            lower_count = len(self.hands['landlord'])
            landlord_pos = 'lower'
        else:  # farmer2
            upper_count = len(self.hands['landlord'])
            lower_count = len(self.hands['farmer1'])
            landlord_pos = 'upper'

        # 构建last_play — 必须将player名转换为'self'/'upper'/'lower'对齐AI引擎
        last_play = None
        if self.last_play is not None:
            mapped_player = self._map_player_to_position(self.last_play_player, player)
            last_play = {
                'player': mapped_player,
                'cards': self.last_play['cards'],
                'type': self.last_play['type'],
                'value': self.last_play['value']
            }

        # 最近的出牌历史
        play_history = []
        for round_plays in self.rounds:
            for play in round_plays:
                if play['type'] != 'pass':
                    play_history.append(play)

        return {
            'my_cards': hand.copy(),
            'my_role': my_role,
            'upper_player_count': upper_count,
            'lower_player_count': lower_count,
            'last_play': last_play,
            'play_history': play_history,
            'landlord': landlord_pos
        }

    def _remove_cards(self, hand: List[str], cards: List[str]) -> None:
        """从手牌中移除卡牌"""
        for card in cards:
            if card in hand:
                hand.remove(card)

    def _next_player(self, current: str) -> str:
        """获取下一个玩家"""
        order = ['landlord', 'farmer1', 'farmer2']
        idx = order.index(current)
        return order[(idx + 1) % 3]

    def _should_play(self, player: str) -> bool:
        """判断玩家是否需要出牌（有能打过上家的牌）"""
        hand = self.hands[player]
        if not hand:
            return False
        if self.last_play is None:
            return True  # 首轮自由出牌

        ai, _ = self._get_ai_and_role(player)
        state = self._build_state(player)
        all_plays = ai._generate_all_plays(hand)

        # 检查是否有能打过的牌
        valid_plays = ai._filter_valid_plays(all_plays, self.last_play)
        return len(valid_plays) > 0

    def _play_turn(self) -> Tuple[bool, Optional[str]]:
        """执行一轮中的一个出牌机会"""
        player = self.current_player
        hand = self.hands[player]

        if not hand:
            # 此人已出完
            return False, None

        ai, my_role = self._get_ai_and_role(player)
        state = self._build_state(player)

        # AI决策
        decision = ai.decide(state)
        cards_to_play = decision['cards']
        play_type = decision['type']

        if play_type == 'pass' or not cards_to_play:
            # Pass
            self.current_round.append({
                'player': player,
                'cards': [],
                'type': 'pass',
                'value': 0
            })
            self.pass_count += 1
        else:
            # 出牌
            self._remove_cards(hand, cards_to_play)

            # 计算value
            state_mgr = GameStateManager()
            ranks = [state_mgr._get_rank(c) for c in cards_to_play]
            max_rank = max(ranks, key=lambda r: state_mgr.CARD_VALUE[r])
            value = state_mgr.CARD_VALUE[max_rank]
            if play_type == 'rocket':
                value = 100
            elif play_type == 'bomb':
                value = 20 + state_mgr.CARD_VALUE[ranks[0]]

            play_record = {
                'player': player,
                'cards': cards_to_play.copy(),
                'type': play_type,
                'value': value
            }

            self.current_round.append(play_record)
            self.last_play = play_record
            self.last_play_player = player
            self.pass_count = 0

            # 检查是否出完了
            if len(hand) == 0:
                return True, player

        # 下一个玩家
        self.current_player = self._next_player(self.current_player)
        return False, None

    def _end_round(self) -> None:
        """结束当前轮"""
        self.rounds.append(self.current_round.copy())
        self.current_round = []

        # 如果连续pass，重置last_play
        if self.pass_count >= 3:
            self.last_play = None
            self.last_play_player = None
            self.pass_count = 0

            # 下一轮从最后出牌的人的下家开始
            if self.last_play_player:
                self.current_player = self._next_player(self.last_play_player)
            # 否则继续从当前玩家出牌（理论上不会到这里）
        else:
            # 继续从当前玩家出牌（他们需要回应上家的牌）
            pass

    def run(self) -> Dict:
        """运行完整游戏"""
        round_num = 0
        max_rounds = 200

        while round_num < max_rounds:
            round_num += 1

            # 一轮：三个玩家各尝试出牌一次
            self.pass_count = 0

            for _ in range(3):
                done, winner = self._play_turn()
                if done:
                    break

                # 如果已经没有人需要出牌（都pass了），可以结束轮
                if self.pass_count >= 2 and self.last_play is None:
                    break

            # 结束轮
            self._end_round()

            if winner:
                break

            # 防止死循环
            if round_num >= max_rounds:
                break

        # 判定胜负
        if winner == 'landlord':
            result = '地主胜'
            winner_name = '地主(平衡策略)'
        else:
            result = '农民胜'
            if winner == 'farmer1':
                winner_name = '农民1(激进策略)'
            else:
                winner_name = '农民2(保守策略)'

        return {
            'result': result,
            'winner': winner_name,
            'winner_hand': winner,
            'rounds': round_num,
            'play_history': [p for round_plays in self.rounds for p in round_plays]
        }


def run_simulation():
    """运行一次博弈模拟"""
    game = DoudizhuGame()
    deal_result = game.deal()

    print("=" * 70)
    print("斗地主AI博弈模拟 - 裁判Agent")
    print("=" * 70)

    # 显示发牌
    print("\n【发牌结果】")
    print(f"  地主(平衡策略): {format_cards(deal_result['landlord'])}")
    print(f"  农民1(激进策略): {format_cards(deal_result['farmer1'])}")
    print(f"  农民2(保守策略): {format_cards(deal_result['farmer2'])}")
    print(f"  底牌: {format_cards(deal_result['bottom'])}")

    # 运行游戏
    result = game.run()

    print("\n【博弈过程】")
    for i, play in enumerate(result['play_history'], 1):
        player_name = {
            'landlord': '地主(平衡)',
            'farmer1': '农民1(激进)',
            'farmer2': '农民2(保守)'
        }.get(play['player'], play['player'])

        if play['type'] == 'pass':
            print(f"  回合{i}: {player_name} -> 过")
        else:
            print(f"  回合{i}: {player_name} -> {play['type']} {format_cards(play['cards'])}")

    print(f"\n【结果】{result['result']}")
    print(f"  获胜方: {result['winner']}")
    print(f"  出牌轮数: {result['rounds']}")

    # 分析
    print("\n【分析】")
    if result['result'] == '地主胜':
        print(f"  地主(平衡策略)获胜。")
        print(f"  地主手牌数量优势加上底牌加成，在博弈中占据上风。")
    else:
        print(f"  农民(激进+保守策略)获胜。")
        print(f"  两名农民通过配合，成功压制了地主的出牌。")

    return result


if __name__ == "__main__":
    run_simulation()