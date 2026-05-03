"""
斗地主策略对比测试 - 简化输出版本
运行多局博弈，比较激进、保守、平衡三种策略的表现
"""

import sys
import random
import copy
from collections import Counter, defaultdict

# 禁用调试输出
sys.path.insert(0, 'modules')
from ai_engine import DoudizhuAI

# 关闭AI调试输出
DoudizhuAI.DEBUG_OUTPUT = False


# ============================================================
# 斗地主游戏引擎
# ============================================================

class DoudizhuGame:
    """斗地主游戏引擎"""

    FULL_DECK = [
        '3S', '3H', '3D', '3C',
        '4S', '4H', '4D', '4C',
        '5S', '5H', '5D', '5C',
        '6S', '6H', '6D', '6C',
        '7S', '7H', '7D', '7C',
        '8S', '8H', '8D', '8C',
        '9S', '9H', '9D', '9C',
        '10S', '10H', '10D', '10C',
        'JS', 'JH', 'JD', 'JC',
        'QS', 'QH', 'QD', 'QC',
        'KS', 'KH', 'KD', 'KC',
        'AS', 'AH', 'AD', 'AC',
        '2S', '2H', '2D', '2C',
        'JB', 'JR'
    ]

    CARD_VALUE = {
        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17, 'JB': 20, 'JR': 30
    }

    def __init__(self, landlord_strategy='aggressive', farmer1_strategy='conservative', farmer2_strategy='balanced'):
        self.landlord_strategy = landlord_strategy
        self.farmer1_strategy = farmer1_strategy
        self.farmer2_strategy = farmer2_strategy

        # 创建各策略AI
        self.ais = {
            'landlord': self._create_ai(landlord_strategy),
            'farmer1': self._create_ai(farmer1_strategy),
            'farmer2': self._create_ai(farmer2_strategy)
        }

    def _create_ai(self, strategy):
        """根据策略名称创建AI"""
        ai = DoudizhuAI()

        # 调整AI策略参数
        if strategy == 'aggressive':
            ai.bomb_aggressive = 0.9
            ai.rocket_reserve = 0.2
            ai.small_card_lead = 0.2
            ai.early_game_penalty = 5
        elif strategy == 'conservative':
            ai.bomb_aggressive = 0.2
            ai.rocket_reserve = 0.9
            ai.small_card_lead = 0.8
            ai.early_game_penalty = 15
        elif strategy == 'balanced':
            ai.bomb_aggressive = 0.5
            ai.rocket_reserve = 0.5
            ai.small_card_lead = 0.5
            ai.early_game_penalty = 10

        return ai

    @staticmethod
    def _card_sort_key(card):
        """牌排序key"""
        rank = card[0:-1]
        return DoudizhuGame.CARD_VALUE.get(rank, 0)

    @staticmethod
    def _identify_play_type(cards):
        """识别出牌类型"""
        if len(cards) == 0:
            return {'type': 'pass', 'cards': [], 'value': 0}

        if len(cards) == 1:
            rank = cards[0][0:-1]
            return {'type': 'single', 'cards': cards, 'value': DoudizhuGame.CARD_VALUE.get(rank, 0)}

        if len(cards) == 2:
            if cards[0][0:-1] == cards[1][0:-1]:
                rank = cards[0][0:-1]
                return {'type': 'pair', 'cards': cards, 'value': DoudizhuGame.CARD_VALUE.get(rank, 0)}
            if set(cards) == {'JB', 'JR'}:
                return {'type': 'rocket', 'cards': cards, 'value': 1000}

        if len(cards) == 3:
            if cards[0][0:-1] == cards[1][0:-1] == cards[2][0:-1]:
                rank = cards[0][0:-1]
                return {'type': 'triple', 'cards': cards, 'value': DoudizhuGame.CARD_VALUE.get(rank, 0)}

        if len(cards) == 4:
            ranks = [c[0:-1] for c in cards]
            counter = Counter(ranks)
            most_common = counter.most_common(1)[0]
            if most_common[1] == 3:
                return {'type': 'triple_with_single', 'cards': cards,
                        'value': DoudizhuGame.CARD_VALUE.get(most_common[0], 0)}
            if most_common[1] == 4:
                rank = most_common[0]
                return {'type': 'bomb', 'cards': cards, 'value': DoudizhuGame.CARD_VALUE.get(rank, 0) + 100}

        if len(cards) >= 5:
            ranks = [c[0:-1] for c in cards]
            rank_order = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            try:
                indices = [rank_order.index(r) for r in ranks]
                indices.sort()
                is_straight = all(indices[i+1] - indices[i] == 1 for i in range(len(indices)-1))
                if is_straight and len(set(ranks)) == len(ranks):
                    return {'type': 'straight', 'cards': cards, 'value': indices[-1]}
            except:
                pass

        return {'type': 'unknown', 'cards': cards, 'value': 0}

    @staticmethod
    def deal_cards():
        """发牌"""
        deck = copy.copy(DoudizhuGame.FULL_DECK)
        random.shuffle(deck)

        farmer1 = sorted(deck[0:17], key=lambda x: DoudizhuGame._card_sort_key(x))
        farmer2 = sorted(deck[17:34], key=lambda x: DoudizhuGame._card_sort_key(x))
        landlord = sorted(deck[34:51], key=lambda x: DoudizhuGame._card_sort_key(x))

        return landlord, farmer1, farmer2

    def play_game(self):
        """运行一局游戏"""
        landlord_hand, farmer1_hand, farmer2_hand = self.deal_cards()

        current_player = 'landlord'
        last_play = None
        last_player = None
        round_num = 0

        hand_sizes = {'landlord': len(landlord_hand), 'farmer1': len(farmer1_hand), 'farmer2': len(farmer2_hand)}

        while True:
            # 检查胜利条件
            if len(landlord_hand) == 0:
                return 'landlord', round_num
            if len(farmer1_hand) == 0 or len(farmer2_hand) == 0:
                return 'farmers', round_num

            # 获取当前玩家手牌
            if current_player == 'landlord':
                current_hand = landlord_hand
            elif current_player == 'farmer1':
                current_hand = farmer1_hand
            else:
                current_hand = farmer2_hand

            # 确定对手和伙伴数量
            if current_player == 'landlord':
                opponent_count = hand_sizes['farmer1']
                partner_count = hand_sizes['farmer2']
            elif current_player == 'farmer1':
                opponent_count = hand_sizes['farmer2']
                partner_count = hand_sizes['landlord']
            else:
                opponent_count = hand_sizes['landlord']
                partner_count = hand_sizes['farmer1']

            # 构造状态
            state_last_play = None
            if last_play is not None and last_player != current_player:
                state_last_play = {
                    'player': last_player,
                    'type': last_play['type'],
                    'cards': last_play['cards'],
                    'value': last_play['value']
                }

            my_role = 'landlord' if current_player == 'landlord' else 'farmer'

            state = {
                'my_cards': current_hand,
                'my_role': my_role,
                'last_play': state_last_play,
                'upper_player_count': opponent_count,
                'lower_player_count': partner_count,
                'upper_last': last_play['cards'] if last_play and last_player != current_player else [],
                'lower_last': [],
            }

            # AI决策
            ai = self.ais[current_player]
            decision = ai.decide(state)

            if decision['action'] == 'pass' or decision['type'] == 'pass':
                play_cards = []
                play_info = {'type': 'pass', 'cards': [], 'value': 0}
            else:
                play_cards = decision['cards']
                play_info = self._identify_play_type(play_cards)

            # 更新手牌
            if play_cards:
                for card in play_cards:
                    if card in current_hand:
                        current_hand.remove(card)
                last_play = play_info
                last_player = current_player
            else:
                if last_play is None or last_play['type'] == 'pass':
                    last_play = None

            hand_sizes[current_player] = len(current_hand)

            # 轮转玩家
            if current_player == 'landlord':
                current_player = 'farmer1'
            elif current_player == 'farmer1':
                current_player = 'farmer2'
            else:
                current_player = 'landlord'

            # 重置出牌
            if play_cards == [] and last_player is None:
                last_play = None

            round_num += 1
            if round_num > 500:
                return 'draw', round_num


def run_tournament(n_games=10):
    """运行策略对比联赛"""
    print("=" * 60)
    print("==> Dou dizhu Strategy Comparison Tournament")
    print("=" * 60)
    print(f"Running {n_games} games per configuration\n")

    # 统计：key = (landlord_strategy, winner) -> count
    stats = defaultdict(int)
    total_rounds = []

    # 测试配置：轮流做地主
    configs = [
        ('aggressive', 'conservative', 'balanced'),
        ('balanced', 'aggressive', 'conservative'),
        ('conservative', 'balanced', 'aggressive'),
    ]

    for landlord_strat, f1_strat, f2_strat in configs:
        print(f"Landlord={landlord_strat}, Farmer1={f1_strat}, Farmer2={f2_strat}")

        for i in range(n_games):
            game = DoudizhuGame(landlord_strat, f1_strat, f2_strat)
            winner, rounds = game.play_game()
            stats[(landlord_strat, winner)] += 1
            total_rounds.append(rounds)

            if (i + 1) % 5 == 0:
                landlord_wins = sum(stats[(landlord_strat, 'landlord')] for _ in [1])
                actual_wins = 0
                for (ls, w), c in stats.items():
                    if ls == landlord_strat and w == 'landlord':
                        actual_wins += c
                print(f"  {i+1}/{n_games} games - Landlord win rate: {actual_wins/(i+1):.1%}")

    # 打印最终结果
    print("\n" + "=" * 60)
    print("==> Final Results")
    print("=" * 60)

    # 按地主策略分组统计
    results_by_strat = {}
    for strat in ['aggressive', 'conservative', 'balanced']:
        landlord_wins = 0
        farmers_wins = 0
        for (ls, w), c in stats.items():
            if ls == strat:
                if w == 'landlord':
                    landlord_wins += c
                else:
                    farmers_wins += c

        total = landlord_wins + farmers_wins
        if total > 0:
            win_rate = landlord_wins / total
            results_by_strat[strat] = {'wins': landlord_wins, 'total': total, 'win_rate': win_rate}
            print(f"\n{strat.upper()} as Landlord:")
            print(f"  Total games: {total}")
            print(f"  Landlord win rate: {win_rate:.1%} ({landlord_wins} wins)")
            print(f"  Farmers win rate: {1-win_rate:.1%} ({farmers_wins} wins)")

    # 计算平均回合数
    avg_rounds = sum(total_rounds) / len(total_rounds) if total_rounds else 0
    print(f"\nAverage rounds per game: {avg_rounds:.1f}")

    # 找出最佳策略
    best_strat = None
    best_win_rate = 0
    for strat, data in results_by_strat.items():
        if data['win_rate'] > best_win_rate:
            best_win_rate = data['win_rate']
            best_strat = strat

    print(f"\nBest Landlord Strategy: {best_strat.upper()} (Win Rate: {best_win_rate:.1%})")

    return stats, results_by_strat


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_tournament(n_games)