"""
斗地主多Agent自我对弈学习系统 v2
通过多Agent对战和遗传算法优化出牌策略

架构：
- StrategyGenome: 策略基因组，编码AI的策略参数
- GeneticOptimizer: 遗传算法优化器
- SelfPlayArena: 自我对弈竞技场
- LearningDashboard: 学习仪表板

使用方法：
    python multi_agent_game.py learn [代数]   # 开始学习
    python multi_agent_game.py battle         # 对战测试
    python multi_agent_game.py status         # 查看状态
"""

import sys
import random
import copy
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

sys.path.insert(0, 'modules')
from ai_engine import DoudizhuAI


# ============================================================
# 基因组感知型AI - 将策略参数应用到评分函数
# ============================================================

class GenomicDoudizhuAI(DoudizhuAI):
    """使用策略基因组的AI - 将基因组参数映射到评分函数"""

    def __init__(self, genome: StrategyGenome = None):
        super().__init__()
        self.genome = genome or StrategyGenome()
        # 禁用调试输出以减少学习时的噪音
        DoudizhuAI.DEBUG_OUTPUT = False

    def _evaluate_play(self, play: Dict, state: Dict) -> float:
        """使用基因组参数评估出牌"""
        g = self.genome  # 简化访问
        score = 0.0
        play_type = play['type']
        my_cards = state.get('my_cards', [])
        my_role = state.get('my_role', 'farmer')
        remaining_my = len(my_cards) - len(play['cards'])
        upper_count = state.get('upper_player_count', 0)
        lower_count = state.get('lower_player_count', 0)
        last_play_info = state.get('last_play')

        # 游戏阶段
        if remaining_my > 10:
            game_phase = 'early'
        elif remaining_my >= 5:
            game_phase = 'mid'
        else:
            game_phase = 'late'

        # === 出牌效率分 ===
        if remaining_my == 0:
            score += 2000
        else:
            efficiency = len(play['cards']) / len(my_cards)
            score += efficiency * 100 * g.efficiency_weight

        # === 牌型基础分（使用基因组的权重） ===
        type_scores = {
            'single': 1 * g.type_weight,
            'pair': 2 * g.type_weight,
            'triple': 3 * g.type_weight,
            'triple_with_single': 4 * g.type_weight,
            'triple_with_pair': 5 * g.type_weight,
            'straight': g.straight_base_score,
            'consecutive_pairs': 6 * g.type_weight,
            'airplane': 8 * g.type_weight,
            'airplane_with_singles': 9 * g.type_weight,
            'airplane_with_pairs': 10 * g.type_weight,
            'bomb': g.bomb_base_score,
            'four_with_two_singles': 10 * g.type_weight,
            'four_with_two_pairs': 12 * g.type_weight,
            'rocket': g.rocket_base_score
        }
        score += type_scores.get(play_type, 0)

        # === 最小能打过原则 ===
        if last_play_info and last_play_info.get('type') != 'pass' and last_play_info.get('cards'):
            last_type = last_play_info.get('type', '')
            last_value = last_play_info.get('value', 0)

            if play_type in ('bomb', 'rocket') and last_type not in ('bomb', 'rocket'):
                if play_type == 'bomb':
                    score -= 50 * g.bomb_aggressive
                elif play_type == 'rocket':
                    score -= 80 * g.rocket_reserve
            elif last_type == 'bomb' and play_type == 'rocket':
                score += 20
            elif play_type == last_type:
                play_value = play.get('value', 0)
                overkill = play_value - last_value
                if overkill > 0:
                    score += 10 - overkill * 0.5

        # === 自由出牌策略 ===
        if not last_play_info or last_play_info.get('type') == 'pass' or not last_play_info.get('cards'):
            if play_type == 'rocket':
                score -= 80 * g.rocket_reserve
            elif play_type == 'bomb':
                score -= 40 * g.bomb_aggressive

            if play_type == 'single':
                val = play.get('value', 0)
                if val <= 6:
                    score += 8 * g.small_card_lead
                elif val == 17:
                    score -= 15
                elif val >= 20:
                    score -= 30 * g.rocket_reserve
            elif play_type == 'pair':
                val = play.get('value', 0)
                if val <= 6:
                    score += 5 * g.small_card_lead
                elif val >= 17:
                    score -= 10
            elif play_type == 'straight':
                score += 8
            elif play_type == 'consecutive_pairs':
                score += 6

        # === 角色策略 ===
        if my_role == 'landlord':
            if play_type not in ['bomb', 'rocket']:
                env_val = play.get('value', 0)
                if env_val < 10:
                    score += g.landlord_bomb_bonus
                elif env_val >= 14:
                    score -= 3
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
                        score += g.farmer_bomb_bonus

        # === 炸弹/火箭保留策略 ===
        if play_type == 'bomb':
            if upper_count <= g.opponent_low_threshold or lower_count <= g.opponent_low_threshold:
                score += 30 * g.bomb_aggressive
            else:
                score -= 15
        if play_type == 'rocket':
            if upper_count <= g.opponent_low_threshold or lower_count <= g.opponent_low_threshold:
                score += 40 * g.rocket_reserve
            else:
                score -= 30 * g.rocket_reserve

        # === 游戏阶段感知 ===
        if game_phase == 'early':
            if play_type == 'bomb':
                score -= g.early_game_penalty
            if play_type in ('single', 'pair') and play.get('value', 0) >= 14:
                score += 3
        elif game_phase == 'late':
            if play_type == 'bomb':
                score += g.late_game_bonus
            if remaining_my <= 3:
                score += 15

        # === 链式思考（基因组权重） ===
        if g.chain_weight > 0:
            # 评估剩余手牌结构
            remaining_cards = my_cards[:-len(play['cards'])] if len(play['cards']) <= len(my_cards) else []
            if remaining_cards:
                ranks = [c[0:-1] for c in remaining_cards if not c.startswith('Joker')]
                counter = Counter(ranks)
                pair_count = sum(1 for r, c in counter.items() if c >= 2)
                triple_count = sum(1 for r, c in counter.items() if c >= 3)
                structure_score = pair_count * 2 + triple_count * 3 - len(remaining_cards) * 0.5
                score += structure_score * g.chain_weight

        return score


# ============================================================
# 策略基因组 - 编码AI的策略参数
# ============================================================

@dataclass
class StrategyGenome:
    """策略基因组 - 代表一组完整的AI策略参数"""

    # 基础分数权重
    efficiency_weight: float = 1.0       # 效率权重
    type_weight: float = 1.0             # 牌型权重
    chain_weight: float = 0.3            # 链式思考权重

    # 特殊牌型分数
    bomb_base_score: float = 15.0        # 炸弹基础分
    rocket_base_score: float = 20.0      # 火箭基础分
    straight_base_score: float = 4.0     # 顺子基础分

    # 行为偏好
    bomb_aggressive: float = 0.5         # 炸弹激进程度 [0-1]
    rocket_reserve: float = 0.5          # 火箭保留程度 [0-1]
    small_card_lead: float = 0.5         # 小牌先手倾向 [0-1]

    # 角色策略
    landlord_bomb_bonus: float = 5.0     # 地主炸弹加成
    farmer_bomb_bonus: float = 5.0       # 农民炸弹加成

    # 游戏阶段感知
    early_game_penalty: float = 10.0     # 早游戏炸弹惩罚
    late_game_bonus: float = 5.0         # 晚期游戏加成

    # 关键阈值
    opponent_low_threshold: int = 4       # 对手手牌少时使用炸弹的阈值
    bomb_value_threshold: float = 0.5    # 使用炸弹的价值阈值

    # 标识
    name: str = "default"
    generation: int = 0
    fitness: float = 0.0  # 适应度（胜率）

    def mutate(self, rate: float = 0.1) -> 'StrategyGenome':
        """基因变异"""
        child = StrategyGenome(
            efficiency_weight=self.efficiency_weight,
            type_weight=self.type_weight,
            chain_weight=self.chain_weight,
            bomb_base_score=self.bomb_base_score,
            rocket_base_score=self.rocket_base_score,
            straight_base_score=self.straight_base_score,
            bomb_aggressive=self.bomb_aggressive,
            rocket_reserve=self.rocket_reserve,
            small_card_lead=self.small_card_lead,
            landlord_bomb_bonus=self.landlord_bomb_bonus,
            farmer_bomb_bonus=self.farmer_bomb_bonus,
            early_game_penalty=self.early_game_penalty,
            late_game_bonus=self.late_game_bonus,
            opponent_low_threshold=self.opponent_low_threshold,
            bomb_value_threshold=self.bomb_value_threshold,
            name=f"{self.name}_m{random.randint(1000,9999)}",
            generation=self.generation + 1,
            fitness=0.0
        )

        # 变异各个参数
        for attr in ['efficiency_weight', 'type_weight', 'chain_weight',
                     'bomb_base_score', 'rocket_base_score', 'straight_base_score',
                     'bomb_aggressive', 'rocket_reserve', 'small_card_lead',
                     'landlord_bomb_bonus', 'farmer_bomb_bonus',
                     'early_game_penalty', 'late_game_bonus']:
            if random.random() < rate:
                current = getattr(child, attr)
                delta = current * 0.3 * random.choice([-1, 1])
                new_val = max(0.1, current + delta)
                setattr(child, attr, new_val)

        # 整数参数变异
        if random.random() < rate:
            child.opponent_low_threshold = random.randint(2, 8)

        return child

    @staticmethod
    def crossover(parent1: 'StrategyGenome', parent2: 'StrategyGenome') -> 'StrategyGenome':
        """基因交叉"""
        child = StrategyGenome(
            efficiency_weight=random.choice([parent1.efficiency_weight, parent2.efficiency_weight]),
            type_weight=random.choice([parent1.type_weight, parent2.type_weight]),
            chain_weight=random.choice([parent1.chain_weight, parent2.chain_weight]),
            bomb_base_score=random.choice([parent1.bomb_base_score, parent2.bomb_base_score]),
            rocket_base_score=random.choice([parent1.rocket_base_score, parent2.rocket_base_score]),
            straight_base_score=random.choice([parent1.straight_base_score, parent2.straight_base_score]),
            bomb_aggressive=random.choice([parent1.bomb_aggressive, parent2.bomb_aggressive]),
            rocket_reserve=random.choice([parent1.rocket_reserve, parent2.rocket_reserve]),
            small_card_lead=random.choice([parent1.small_card_lead, parent2.small_card_lead]),
            landlord_bomb_bonus=random.choice([parent1.landlord_bomb_bonus, parent2.landlord_bomb_bonus]),
            farmer_bomb_bonus=random.choice([parent1.farmer_bomb_bonus, parent2.farmer_bomb_bonus]),
            early_game_penalty=random.choice([parent1.early_game_penalty, parent2.early_game_penalty]),
            late_game_bonus=random.choice([parent1.late_game_bonus, parent2.late_game_bonus]),
            opponent_low_threshold=random.choice([parent1.opponent_low_threshold, parent2.opponent_low_threshold]),
            bomb_value_threshold=random.choice([parent1.bomb_value_threshold, parent2.bomb_value_threshold]),
            name=f"{parent1.name[:4]}_{parent2.name[:4]}_c{random.randint(1000,9999)}",
            generation=max(parent1.generation, parent2.generation) + 1,
            fitness=0.0
        )
        return child

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'efficiency_weight': self.efficiency_weight,
            'type_weight': self.type_weight,
            'chain_weight': self.chain_weight,
            'bomb_base_score': self.bomb_base_score,
            'rocket_base_score': self.rocket_base_score,
            'straight_base_score': self.straight_base_score,
            'bomb_aggressive': self.bomb_aggressive,
            'rocket_reserve': self.rocket_reserve,
            'small_card_lead': self.small_card_lead,
            'landlord_bomb_bonus': self.landlord_bomb_bonus,
            'farmer_bomb_bonus': self.farmer_bomb_bonus,
            'early_game_penalty': self.early_game_penalty,
            'late_game_bonus': self.late_game_bonus,
            'opponent_low_threshold': self.opponent_low_threshold,
            'bomb_value_threshold': self.bomb_value_threshold,
            'name': self.name,
            'generation': self.generation,
            'fitness': self.fitness
        }

    @staticmethod
    def from_dict(d: Dict) -> 'StrategyGenome':
        """从字典创建"""
        return StrategyGenome(
            efficiency_weight=d.get('efficiency_weight', 1.0),
            type_weight=d.get('type_weight', 1.0),
            chain_weight=d.get('chain_weight', 0.3),
            bomb_base_score=d.get('bomb_base_score', 15.0),
            rocket_base_score=d.get('rocket_base_score', 20.0),
            straight_base_score=d.get('straight_base_score', 4.0),
            bomb_aggressive=d.get('bomb_aggressive', 0.5),
            rocket_reserve=d.get('rocket_reserve', 0.5),
            small_card_lead=d.get('small_card_lead', 0.5),
            landlord_bomb_bonus=d.get('landlord_bomb_bonus', 5.0),
            farmer_bomb_bonus=d.get('farmer_bomb_bonus', 5.0),
            early_game_penalty=d.get('early_game_penalty', 10.0),
            late_game_bonus=d.get('late_game_bonus', 5.0),
            opponent_low_threshold=d.get('opponent_low_threshold', 4),
            bomb_value_threshold=d.get('bomb_value_threshold', 0.5),
            name=d.get('name', 'default'),
            generation=d.get('generation', 0),
            fitness=d.get('fitness', 0.0)
        )


# ============================================================
# 策略优化器 - 管理基因组池和遗传算法
# ============================================================

class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self, population_size: int = 10, elite_ratio: float = 0.3):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.history: List[Dict] = []

        # 初始化种群
        self._init_population()

    def _init_population(self):
        """初始化种群"""
        base_genome = StrategyGenome(name="balanced_v1", generation=0)

        # 创建不同风格的初始个体
        strategies = [
            StrategyGenome(name="aggressive", generation=0,
                          bomb_aggressive=0.8, rocket_reserve=0.3, small_card_lead=0.3),
            StrategyGenome(name="conservative", generation=0,
                          bomb_aggressive=0.3, rocket_reserve=0.8, small_card_lead=0.7),
            StrategyGenome(name="balanced", generation=0),
            StrategyGenome(name="chain_master", generation=0,
                          chain_weight=0.6, efficiency_weight=1.2),
            StrategyGenome(name="bomb_lover", generation=0,
                          bomb_base_score=20.0, bomb_aggressive=0.9),
        ]

        self.population = strategies[:self.population_size]

        # 如果不够，补全随机个体
        while len(self.population) < self.population_size:
            g = StrategyGenome(name=f"random_{len(self.population)}", generation=0)
            self.population.append(g)

    def evaluate_fitness(self, genome: StrategyGenome, games: int = 10) -> float:
        """评估基因组的适应度（胜率）"""
        wins = 0
        total = games

        for _ in range(games):
            result = SelfPlayArena.run_single_game_with_genome(genome)
            if result == 'landlord':
                wins += 1

        win_rate = wins / total
        genome.fitness = win_rate
        return win_rate

    def evolve(self) -> List[StrategyGenome]:
        """进化一代"""
        # 按适应度排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # 记录历史
        best = self.population[0]
        avg = sum(g.fitness for g in self.population) / len(self.population)
        self.history.append({
            'generation': self.generation,
            'best_fitness': best.fitness,
            'avg_fitness': avg,
            'best_name': best.name
        })

        # 保留精英
        elite_count = max(2, int(self.population_size * self.elite_ratio))
        elites = self.population[:elite_count]

        # 生成新一代
        new_population = list(elites)  # 保留精英

        while len(new_population) < self.population_size:
            # 选择父代（锦标赛选择）
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # 交叉
            if random.random() < 0.7:
                child = StrategyGenome.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            # 变异
            child = child.mutate(rate=0.15)

            new_population.append(child)

        self.population = new_population[:self.population_size]
        self.generation += 1

        return elites

    def _tournament_select(self, tournament_size: int = 3) -> StrategyGenome:
        """锦标赛选择"""
        candidates = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def get_best(self) -> StrategyGenome:
        """获取最佳个体"""
        return max(self.population, key=lambda x: x.fitness)

    def save(self, path: str):
        """保存种群状态"""
        data = {
            'population': [g.to_dict() for g in self.population],
            'generation': self.generation,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """加载种群状态"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.population = [StrategyGenome.from_dict(g) for g in data['population']]
        self.generation = data['generation']
        self.history = data.get('history', [])


# ============================================================
# 自我对弈竞技场
# ============================================================

class SelfPlayArena:
    """自我对弈竞技场 - 使用策略基因组进行对局"""

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

    def __init__(self):
        pass

    @staticmethod
    def _card_sort_key(card: str) -> int:
        """牌排序key"""
        rank = card[0:-1]
        rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17,
                    'JB': 20, 'JR': 30}
        return rank_map.get(rank, 0)

    @staticmethod
    def _identify_play_type(cards: List[str]) -> Dict:
        """识别出牌类型"""
        if len(cards) == 0:
            return {'type': 'pass', 'cards': [], 'value': 0}

        if len(cards) == 1:
            rank = cards[0][0:-1]
            rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                        '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17,
                        'JB': 20, 'JR': 30}
            return {'type': 'single', 'cards': cards, 'value': rank_map.get(rank, 0)}

        if len(cards) == 2 and cards[0][0:-1] == cards[1][0:-1]:
            rank = cards[0][0:-1]
            rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                        '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17,
                        'JB': 20, 'JR': 30}
            return {'type': 'pair', 'cards': cards, 'value': rank_map.get(rank, 0)}

        if len(cards) == 3 and cards[0][0:-1] == cards[1][0:-1] == cards[2][0:-1]:
            rank = cards[0][0:-1]
            rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                        '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17}
            return {'type': 'triple', 'cards': cards, 'value': rank_map.get(rank, 0)}

        if len(cards) == 4:
            ranks = [c[0:-1] for c in cards]
            counter = Counter(ranks)
            most_common = counter.most_common(1)[0]
            if most_common[1] == 3:
                rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                            '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17}
                return {'type': 'triple_with_single', 'cards': cards,
                        'value': rank_map.get(most_common[0], 0)}

        if len(cards) == 4 and cards[0][0:-1] == cards[1][0:-1] == cards[2][0:-1] == cards[3][0:-1]:
            rank = cards[0][0:-1]
            rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                        '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17}
            return {'type': 'bomb', 'cards': cards, 'value': rank_map.get(rank, 0) + 100}

        if len(cards) == 2 and set(cards) == {'JB', 'JR'}:
            return {'type': 'rocket', 'cards': cards, 'value': 1000}

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
    def deal_cards() -> Tuple[List[str], List[str], List[str], List[str]]:
        """发牌"""
        deck = copy.copy(SelfPlayArena.FULL_DECK)
        random.shuffle(deck)

        farmer1 = sorted(deck[0:17], key=lambda x: SelfPlayArena._card_sort_key(x))
        farmer2 = sorted(deck[17:34], key=lambda x: SelfPlayArena._card_sort_key(x))
        landlord = sorted(deck[34:51], key=lambda x: SelfPlayArena._card_sort_key(x))
        bottom = deck[51:54]

        return landlord, farmer1, farmer2, bottom

    @staticmethod
    def run_single_game_with_genome(genome: StrategyGenome) -> str:
        """使用指定基因组运行单局游戏"""
        # 使用基因组AI
        ai = GenomicDoudizhuAI(genome)
        landlord_hand, farmer1_hand, farmer2_hand, bottom = SelfPlayArena.deal_cards()

        current_player = 'landlord'
        last_play = None
        last_player = None
        round_num = 0
        hand_sizes = {'landlord': len(landlord_hand), 'farmer1': len(farmer1_hand), 'farmer2': len(farmer2_hand)}

        while True:
            if len(landlord_hand) == 0:
                return 'landlord'
            if len(farmer1_hand) == 0 or len(farmer2_hand) == 0:
                return 'farmers'

            if current_player == 'landlord':
                current_hand = landlord_hand
            elif current_player == 'farmer1':
                current_hand = farmer1_hand
            else:
                current_hand = farmer2_hand

            opponent_count = hand_sizes['farmer1'] if current_player == 'landlord' else hand_sizes['landlord']
            partner_count = hand_sizes['farmer2'] if current_player in ('landlord', 'farmer1') else hand_sizes['farmer1']

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

            decision = ai.decide(state)

            if decision['action'] == 'pass' or decision['type'] == 'pass':
                play_cards = []
                play_info = {'type': 'pass', 'cards': [], 'value': 0}
            else:
                play_cards = decision['cards']
                play_info = SelfPlayArena._identify_play_type(play_cards)

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

            if current_player == 'landlord':
                current_player = 'farmer1'
            elif current_player == 'farmer1':
                current_player = 'farmer2'
            else:
                current_player = 'landlord'

            if play_cards == [] and last_player is None:
                last_play = None

            round_num += 1
            if round_num > 500:
                return 'draw'


# ============================================================
# 学习仪表板
# ============================================================

class LearningDashboard:
    """学习仪表板 - 展示学习进度和结果"""

    def __init__(self, optimizer: GeneticOptimizer):
        self.optimizer = optimizer
        self.save_path = "learning_state.json"

    def print_status(self):
        """打印当前状态"""
        print("\n" + "=" * 70)
        print("==> 斗地主AI自我对弈学习系统 - 状态面板")
        print("=" * 70)

        print(f"\n[STAT] 当前代数: {self.optimizer.generation}")
        print(f"[GRAPH] 种群大小: {len(self.optimizer.population)}")

        if self.optimizer.history:
            print(f"\n[GRAPH] 历史最佳:")
            for h in self.optimizer.history[-5:]:
                print(f"   Gen {h['generation']}: Best={h['best_fitness']:.1%}, Avg={h['avg_fitness']:.1%} ({h['best_name']})")

        print(f"\n[TOP] 当前种群排名:")
        sorted_pop = sorted(self.optimizer.population, key=lambda x: x.fitness, reverse=True)
        for i, g in enumerate(sorted_pop[:5]):
            print(f"   {i+1}. {g.name}: 胜率={g.fitness:.1%}")

        if sorted_pop:
            best = sorted_pop[0]
            print(f"\n[BEST] 最佳策略详情 ({best.name}):")
            print(f"   炸弹激进度: {best.bomb_aggressive:.2f}")
            print(f"   火箭保留度: {best.rocket_reserve:.2f}")
            print(f"   链式权重: {best.chain_weight:.2f}")
            print(f"   小牌先手: {best.small_card_lead:.2f}")
            print(f"   早游戏惩罚: {best.early_game_penalty:.1f}")

    def save(self):
        """保存状态"""
        self.optimizer.save(self.save_path)
        print(f"\n[SAVE] 状态已保存到 {self.save_path}")

    def load(self):
        """加载状态"""
        if os.path.exists(self.save_path):
            self.optimizer.load(self.save_path)
            print(f"\n[LOAD] 已加载保存的状态")
            return True
        return False


# ============================================================
# 主学习循环
# ============================================================

def run_learning(generations: int = 10, games_per_gen: int = 20):
    """运行学习循环"""
    print("=" * 70)
    print("==> 斗地主AI自我对弈学习")
    print("=" * 70)
    print(f"配置: {generations}代, 每代{games_per_gen}局")

    optimizer = GeneticOptimizer(population_size=10)
    dashboard = LearningDashboard(optimizer)

    # 尝试加载已有状态
    if dashboard.load():
        print("[LOAD] 继续之前的训练...")

    for gen in range(generations):
        print(f"\n{'='*70}")
        print(f"[LOC] 第 {optimizer.generation + 1} 代学习")
        print(f"{'='*70}")

        # 评估所有个体
        for i, genome in enumerate(optimizer.population):
            print(f"\n评估 {genome.name} ({i+1}/{len(optimizer.population)})...")
            win_rate = optimizer.evaluate_fitness(genome, games=games_per_gen)
            print(f"   胜率: {win_rate:.1%}")

        # 打印当前状态
        dashboard.print_status()

        # 进化
        elites = optimizer.evolve()
        print(f"\n[OK] 第{optimizer.generation}代完成")
        print(f"   最佳: {elites[0].name} (胜率: {elites[0].fitness:.1%})")

        # 保存状态
        dashboard.save()

    print(f"\n{'='*70}")
    print("[DONE] 学习完成!")
    print(f"{'='*70}")

    best = optimizer.get_best()
    print(f"\n最终最佳策略: {best.name}")
    print(f"胜率: {best.fitness:.1%}")
    print("\n策略参数:")
    for k, v in best.to_dict().items():
        if k not in ['name', 'generation', 'fitness']:
            print(f"  {k}: {v}")

    return best


def run_battle(n_games: int = 50):
    """运行对战测试"""
    print("=" * 70)
    print("==>  对战测试")
    print("=" * 70)

    optimizer = GeneticOptimizer()

    # 加载最佳策略
    dashboard = LearningDashboard(optimizer)
    dashboard.load()

    best = optimizer.get_best()
    print(f"\n使用策略: {best.name}")
    print(f"历史胜率: {best.fitness:.1%}")

    # 评估
    print(f"\n运行 {n_games} 局对战...")
    wins = 0
    for i in range(n_games):
        result = SelfPlayArena.run_single_game_with_genome(best)
        if result == 'landlord':
            wins += 1
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{n_games}, 当前胜率: {wins/(i+1):.1%}")

    final_win_rate = wins / n_games
    print(f"\n[STAT] 最终测试结果:")
    print(f"   地主胜率: {final_win_rate:.1%}")
    print(f"   农民胜率: {1-final_win_rate:.1%}")

    return final_win_rate


def show_status():
    """显示状态"""
    optimizer = GeneticOptimizer()
    dashboard = LearningDashboard(optimizer)

    if dashboard.load():
        dashboard.print_status()
    else:
        print("没有找到保存的学习状态。请先运行学习:")
        print("  python multi_agent_game.py learn")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python multi_agent_game.py learn [代数] [每代局数]  # 开始学习")
        print("  python multi_agent_game.py battle [局数]             # 对战测试")
        print("  python multi_agent_game.py status                     # 查看状态")
        sys.exit(1)

    command = sys.argv[1]

    if command == "learn":
        generations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        games_per_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        run_learning(generations, games_per_gen)
    elif command == "battle":
        n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_battle(n_games)
    elif command == "status":
        show_status()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)
