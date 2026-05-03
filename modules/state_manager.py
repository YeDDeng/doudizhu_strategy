"""
Module 2: Game State Manager
Tracks game state, play history, and manages the full 54-card deck.
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple
import itertools

class GameStateManager:
    """Game state management and history tracking."""

    RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    SUITS = ['♠', '♥', '♦', '♣']
    JOKERS = ['Joker_B', 'Joker_R']

    # 与DouZero对齐的牌值映射
    CARD_VALUE = {
        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        '2': 17,
        'Joker_B': 20,  # 小王
        'Joker_R': 30   # 大王
    }

    # 完整54张牌（用于初始化）
    ALL_CARDS = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                  8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11,
                  12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 17, 17, 17, 17, 20, 30]

    def __init__(self):
        """Initialize game state manager."""
        self.reset()

    def reset(self) -> None:
        """Reset for new game."""
        self.full_deck = self._init_deck()
        self.my_cards: List[str] = []
        self.played_cards: List[str] = []
        self.play_history: List[Dict] = []
        self.upper_player_played: List[List[str]] = []
        self.lower_player_played: List[List[str]] = []
        self.remaining_cards: List[str] = []
        self.landlord: Optional[str] = None  # 'upper', 'lower', 'self'
        self.my_role: Optional[str] = None  # 'landlord', 'farmer'
        self.current_turn: Optional[str] = None
        self.last_play: Optional[Dict] = None
        self._card_counts = Counter()
        # Display state for opponent plays — cleared when area goes empty
        self._upper_display_cards: List[str] = []
        self._lower_display_cards: List[str] = []
        self._upper_empty_frames = 0
        self._lower_empty_frames = 0
        self._my_cards_empty_frames = 0

    def _init_deck(self) -> List[str]:
        """Initialize a full 54-card deck."""
        deck = []
        for rank in self.RANKS:
            for suit in self.SUITS:
                deck.append(f"{rank}{suit}")
        deck.extend(self.JOKERS)
        return deck

    def update_from_recognition(self, recognition_result: Dict) -> None:
        """Update game state from recognition result."""
        # Update my cards (always, even if empty — empty means round ended or misdetection)
        new_my_cards = recognition_result.get("my_cards", [])
        had_cards_before = len(self.my_cards) > 0
        if new_my_cards:
            self.my_cards = sorted(new_my_cards, key=lambda c: self.CARD_VALUE[self._get_rank(c)])
        else:
            self.my_cards = []
        has_cards_now = len(self.my_cards) > 0

        # Round ended: cards disappeared → debounce with 3-frame threshold
        if had_cards_before and not has_cards_now:
            self._my_cards_empty_frames += 1
            if self._my_cards_empty_frames >= 3:
                self.play_history = []
                self.last_play = None
                self.upper_player_played = []
                self.lower_player_played = []
                self.played_cards = []
                self._upper_display_cards = []
                self._lower_display_cards = []
                self._upper_empty_frames = 0
                self._lower_empty_frames = 0
                self._my_cards_empty_frames = 0
        else:
            self._my_cards_empty_frames = 0

        # Track current players' visible cards
        upper_last = recognition_result.get("upper_player_last", [])
        lower_last = recognition_result.get("lower_player_last", [])

        # Detect pass: if player previously had cards but now shows nothing, they passed
        upper_played_now = len(upper_last) > 0
        lower_played_now = len(lower_last) > 0

        # Track opponent play display state (clear when area is empty for 2+ frames)
        if upper_played_now:
            self._upper_display_cards = list(upper_last)
            self._upper_empty_frames = 0
        else:
            self._upper_empty_frames += 1
            if self._upper_empty_frames >= 8:
                self._upper_display_cards = []

        if lower_played_now:
            self._lower_display_cards = list(lower_last)
            self._lower_empty_frames = 0
        else:
            self._lower_empty_frames += 1
            if self._lower_empty_frames >= 8:
                self._lower_display_cards = []

        # Check for new plays (comparing with the LAST recorded play for that player)
        last_upper = self.upper_player_played[-1] if self.upper_player_played else []
        last_lower = self.lower_player_played[-1] if self.lower_player_played else []

        # Upper player played new cards
        if upper_played_now and upper_last != last_upper:
            self._record_play('upper', upper_last)
        # Upper player passed (no cards shown but had cards before and it's a new round)
        elif not upper_played_now and last_upper and self._should_record_pass('upper'):
            self._record_pass('upper')

        # Lower player played new cards
        if lower_played_now and lower_last != last_lower:
            self._record_play('lower', lower_last)
        # Lower player passed
        elif not lower_played_now and last_lower and self._should_record_pass('lower'):
            self._record_pass('lower')

        # Update current turn
        if "current_turn" in recognition_result:
            self.current_turn = recognition_result["current_turn"]

        # Update landlord
        if "landlord" in recognition_result:
            self.landlord = recognition_result["landlord"]
            self._set_my_role()

        # Update remaining card counts
        self._update_remaining()

    def _record_pass(self, player: str) -> None:
        """Record a pass for a player."""
        play_record = {
            'player': player,
            'cards': [],
            'type': 'pass',
            'value': 0
        }
        self.play_history.append(play_record)
        self.last_play = play_record

    def _should_record_pass(self, player: str) -> bool:
        """Check if we should record a pass for this player."""
        # Only record pass if the last play was by a different player or this player's own last play was a pass
        if not self.play_history:
            return False
        last = self.play_history[-1]
        if last['player'] == player:
            # This player just played, wait for next round
            return False
        # Last play was by another player - if that player hasn't passed yet, this is a valid pass
        return True

    def _record_play(self, player: str, cards: List[str]) -> None:
        """Record a play to history."""
        if not cards:
            return

        card_type = self._identify_card_type(cards)
        # Get highest value for play comparison
        if card_type == 'rocket':
            value = 100
        elif card_type == 'bomb':
            ranks = [self._get_rank(c) for c in cards]
            value = 20 + self.CARD_VALUE[ranks[0]]
        elif card_type in ('triple_with_single', 'triple_with_pair', 'airplane',
                           'airplane_with_singles', 'airplane_with_pairs'):
            # 三带一/二、飞机的比较值取决于"主体"（三张部分），不是踢脚
            rank_counts = Counter(self._get_rank(c) for c in cards)
            triple_rank = max(r for r, cnt in rank_counts.items() if cnt >= 3)
            value = self.CARD_VALUE[triple_rank]
        elif card_type in ('four_with_two_singles', 'four_with_two_pairs'):
            # 四带二的比较值取决于四张部分
            rank_counts = Counter(self._get_rank(c) for c in cards)
            quad_rank = next(r for r, cnt in rank_counts.items() if cnt == 4)
            value = 50 + self.CARD_VALUE[quad_rank]
        else:
            ranks = [self._get_rank(c) for c in cards]
            max_rank = max(ranks, key=lambda r: self.CARD_VALUE[r])
            value = self.CARD_VALUE[max_rank]

        play_record = {
            'player': player,
            'cards': cards.copy(),
            'type': card_type,
            'value': value
        }

        self.play_history.append(play_record)
        self.played_cards.extend(cards)
        self.last_play = play_record

        if player == 'upper':
            self.upper_player_played.append(cards)
        elif player == 'lower':
            self.lower_player_played.append(cards)

    def _identify_card_type(self, cards: List[str]) -> str:
        """Identify the type of play."""
        if not cards:
            return 'pass'

        n = len(cards)
        ranks = [self._get_rank(c) for c in cards]
        rank_counts = Counter(ranks)
        unique_ranks = sorted(set(self.CARD_VALUE[r] for r in ranks))

        # Check for rocket (Joker_B + Joker_R)
        if n == 2 and sorted(cards) == sorted(['Joker_B', 'Joker_R']):
            return 'rocket'

        # Check for bomb (four same rank)
        if n == 4 and len(rank_counts) == 1:
            return 'bomb'

        # Single
        if n == 1:
            return 'single'

        # Pair
        if n == 2 and len(rank_counts) == 1:
            return 'pair'

        # Triple
        if n == 3 and len(rank_counts) == 1:
            return 'triple'

        # Triple with single
        if n == 4 and len(rank_counts) == 2:
            counts = sorted(rank_counts.values())
            if counts == [1, 3]:
                return 'triple_with_single'

        # Triple with pair
        if n == 5 and len(rank_counts) == 2:
            counts = sorted(rank_counts.values())
            if counts == [2, 3]:
                return 'triple_with_pair'

        # Straight (5+ consecutive singles)
        if n >= 5 and len(rank_counts) == n:
            if self._is_consecutive(unique_ranks, n) and all(v <= 14 for v in unique_ranks):
                return 'straight'

        # Consecutive pairs (3+ consecutive pairs)
        if n >= 6 and n % 2 == 0:
            pairs_count = n // 2
            if all(c == 2 for c in rank_counts.values()) and len(rank_counts) == pairs_count:
                if self._is_consecutive(unique_ranks, pairs_count) and all(v <= 14 for v in unique_ranks):
                    return 'consecutive_pairs'

        # Four with two singles
        if n == 6 and len(rank_counts) == 3:
            if 4 in rank_counts.values():
                return 'four_with_two_singles'

        # Four with two pairs
        if n == 8 and len(rank_counts) >= 2:
            counts = sorted(rank_counts.values())
            if counts[-1] == 4 and counts[-2] == 2:
                return 'four_with_two_pairs'

        # Airplane (2+ consecutive triples, optionally with wings)
        triples_count = sum(1 for c in rank_counts.values() if c >= 3)
        if triples_count >= 2:
            triple_ranks = sorted(self.CARD_VALUE[r] for r, cnt in rank_counts.items() if cnt >= 3)
            # Airplane cannot include 2 or jokers
            if all(v < 17 for v in triple_ranks) and self._is_consecutive(triple_ranks, triples_count):
                singles_count = sum(1 for c in rank_counts.values() if c == 1)
                pairs_count = sum(1 for c in rank_counts.values() if c == 2)

                if singles_count == triples_count and n == 4 * triples_count:
                    return 'airplane_with_singles'
                elif pairs_count == triples_count and n == 5 * triples_count:
                    return 'airplane_with_pairs'
                elif singles_count == 0 and pairs_count == 0 and n == 3 * triples_count:
                    return 'airplane'

        return 'unknown'

    def _get_rank(self, card: str) -> str:
        """Extract rank from card string."""
        if card.startswith('Joker'):
            return card
        # Handle single-character ranks (3-9, J, Q, K, A)
        if len(card) == 1:
            return card
        # Handle 10 which is two characters
        if card[1] == '0' and len(card) >= 2:
            return '10'
        return card[0]

    def _is_consecutive(self, values: List[int], length: int) -> bool:
        """Check if values are consecutive."""
        if len(values) != length:
            return False
        for i in range(1, len(values)):
            if values[i] != values[i-1] + 1:
                return False
        return True

    def _same_rank(self, cards: List[str]) -> bool:
        """Check if all cards are the same rank."""
        if not cards:
            return False
        ranks = [self._get_rank(c) for c in cards]
        return all(r == ranks[0] for r in ranks)

    def _update_remaining(self) -> None:
        """Update remaining cards calculation."""
        played_set = set(self.played_cards)
        my_set = set(self.my_cards)
        self.remaining_cards = [c for c in self.full_deck if c not in played_set and c not in my_set]

    def get_remaining_cards(self) -> List[str]:
        """Get remaining unknown cards."""
        return self.remaining_cards

    def get_state(self) -> Dict:
        """Get current game state."""
        upper_count = 17 if self.landlord != 'upper' else 20
        lower_count = 17 if self.landlord != 'lower' else 20
        my_count = len(self.my_cards)

        if self.landlord == 'self':
            upper_count = 17
            lower_count = 17

        # Subtract played cards
        upper_played = sum(len(p) for p in self.upper_player_played)
        lower_played = sum(len(p) for p in self.lower_player_played)

        # Return display cards (tracked separately to clear stale plays)
        # Remove round-end handling
        current_last_play = self.last_play

        return {
            'my_cards': self.my_cards.copy(),
            'my_count': len(self.my_cards),
            'upper_player_count': upper_count - upper_played,
            'lower_player_count': lower_count - lower_played,
            'upper_last': list(self._upper_display_cards),
            'lower_last': list(self._lower_display_cards),
            'total_remaining': len(self.remaining_cards),
            'landlord': self.landlord,
            'my_role': self.my_role,
            'current_turn': self.current_turn,
            'last_play': current_last_play,
            'play_history': self.play_history.copy(),
            'remaining_cards': self.remaining_cards.copy()
        }

    def set_landlord(self, player: str) -> None:
        """Set the landlord."""
        self.landlord = player
        self._set_my_role()

    def _set_my_role(self) -> None:
        """Set my role based on landlord position."""
        if self.landlord == 'self':
            self.my_role = 'landlord'
        else:
            self.my_role = 'farmer'

    def set_my_role(self, role: str) -> None:
        """Manually set my role."""
        self.my_role = role

    def get_my_remaining_count(self) -> int:
        """Get number of cards remaining in my hand."""
        return len(self.my_cards)

    def is_my_turn(self) -> bool:
        """Check if it's my turn.
        If current_turn detection not available, assume it's my turn when I have cards.
        """
        if self.current_turn is None:
            # If turn detection not working, assume it's my turn if I have cards
            return len(self.my_cards) > 0
        return self.current_turn == 'self'


if __name__ == "__main__":
    print("Testing GameStateManager...")

    state = GameStateManager()

    # Test deck initialization
    deck = state._init_deck()
    print(f"Deck size: {len(deck)}")
    assert len(deck) == 54, f"Expected 54 cards, got {len(deck)}"
    print("[OK] Deck initialization correct")

    # Test card type identification
    test_cases = [
        (['3♠'], 'single'),
        (['3♠', '3♥'], 'pair'),
        (['3♠', '3♥', '3♦'], 'triple'),
        (['Joker_B', 'Joker_R'], 'rocket'),
        (['3♠', '3♥', '3♦', '3♣'], 'bomb'),
        (['3♠', '3♥', '3♦', '4♠'], 'triple_with_single'),
        (['3♠', '3♥', '3♦', '4♠', '4♥'], 'triple_with_pair'),
    ]

    all_passed = True
    for cards, expected in test_cases:
        detected = state._identify_card_type(cards)
        passed = detected == expected
        all_passed &= passed
        status = "[OK]" if passed else "[FAIL] expected " + expected
        print(f"  {len(cards)} card(s): detected {detected} → {status}")

    if all_passed:
        print("[OK] All basic card type tests passed")
    else:
        print("[FAIL] Some tests failed")

    # Test remaining cards
    state.reset()
    state.my_cards = ['3♠', '3♥', '3♦', '3♣']
    state._record_play('upper', ['4♠', '4♥'])
    state._update_remaining()
    remaining = state.get_remaining_cards()
    print(f"Remaining cards: {len(remaining)} (expected 54 - 4 - 2 = 48)")
    assert len(remaining) == 48, f"Expected 48, got {len(remaining)}"
    print("[OK] Remaining cards calculation correct")

    print("\nAll tests completed!")
