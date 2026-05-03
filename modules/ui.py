"""
Module 5: Floating UI Window
Always-on-top PyQt5 floating window with draggable, translucent background.
Displays real-time AI suggestions.
"""

import sys
import ctypes
from ctypes import wintypes

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSlider
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor

from typing import Dict, Optional

# Windows MSG struct for nativeEvent handling
class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _POINT),
    ]

WM_NCHITTEST = 0x0084
HTCAPTION = 2


class AIFloatingWindow(QWidget):
    """Always-on-top floating AI suggestion window."""

    close_requested = pyqtSignal()
    play_clicked = pyqtSignal()
    pass_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    start_clicked = pyqtSignal()
    landlord_changed = pyqtSignal(str)  # 'self', 'upper', 'lower'

    def __init__(self):
        """Initialize the floating window."""
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Setup UI components."""
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(320, 410)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Title bar
        title_bar = QFrame()
        title_bar.setFixedHeight(32)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 4, 6, 4)
        title_label = QLabel("🤖 斗地主AI助手")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        # Let nativeEvent handle drags through the label
        title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.on_close_clicked)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)
        main_layout.addWidget(title_bar)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #555555;")
        main_layout.addWidget(separator)

        # Status area
        self.status_label = QLabel("状态: 等待游戏...")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        self.status_label.setStyleSheet("color: #cccccc;")
        main_layout.addWidget(self.status_label)

        # My cards display
        self.my_cards_label = QLabel("我的牌: 未识别")
        self.my_cards_label.setFont(QFont("Microsoft YaHei", 9))
        self.my_cards_label.setStyleSheet("color: #cccccc;")
        my_cards_font = QFont("Microsoft YaHei", 9)
        my_cards_font.setBold(True)
        self.my_cards_label.setFont(my_cards_font)
        main_layout.addWidget(self.my_cards_label)

        # Opponent counts
        self.opponent_label = QLabel("上家: ?张  |  下家: ?张")
        self.opponent_label.setFont(QFont("Microsoft YaHei", 9))
        self.opponent_label.setStyleSheet("color: #cccccc;")
        main_layout.addWidget(self.opponent_label)

        # Opponent played cards
        self.opponent_played_label = QLabel("上家出牌: -  |  下家出牌: -")
        self.opponent_played_label.setFont(QFont("Microsoft YaHei", 9))
        self.opponent_played_label.setStyleSheet("color: #ffaaaa;")
        main_layout.addWidget(self.opponent_played_label)

        # Landlord selector
        landlord_frame = QFrame()
        landlord_layout = QHBoxLayout(landlord_frame)
        landlord_layout.setContentsMargins(0, 2, 0, 2)
        landlord_layout.setSpacing(4)
        landlord_label = QLabel("地主:")
        landlord_label.setFont(QFont("Microsoft YaHei", 9))
        landlord_label.setStyleSheet("color: #cccccc;")
        landlord_layout.addWidget(landlord_label)

        self.landlord_self_btn = QPushButton("自己")
        self.landlord_upper_btn = QPushButton("上家")
        self.landlord_lower_btn = QPushButton("下家")
        self._landlord_buttons = {
            'self': self.landlord_self_btn,
            'upper': self.landlord_upper_btn,
            'lower': self.landlord_lower_btn,
        }
        self._current_landlord = None

        btn_style = """
            QPushButton {
                background-color: #444444;
                color: #aaaaaa;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 3px 8px;
                font-family: Microsoft YaHei;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #555555;
                color: #ffffff;
            }
        """
        btn_active_style = """
            QPushButton {
                background-color: #d4880f;
                color: #ffffff;
                border: 1px solid #f0a020;
                border-radius: 4px;
                padding: 3px 8px;
                font-family: Microsoft YaHei;
                font-size: 9pt;
                font-weight: bold;
            }
        """

        for player, btn in self._landlord_buttons.items():
            btn.setStyleSheet(btn_style)
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda checked, p=player: self._on_landlord_clicked(p))
            landlord_layout.addWidget(btn)

        self._landlord_btn_style = btn_style
        self._landlord_btn_active_style = btn_active_style
        landlord_layout.addStretch()
        main_layout.addWidget(landlord_frame)

        # AI suggestion header
        suggestion_header = QLabel("💡 AI建议")
        suggestion_header.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        suggestion_header.setStyleSheet("color: #88ff88;")
        main_layout.addWidget(suggestion_header)

        # Suggestion area
        self.suggestion_label = QLabel("等待中...")
        self.suggestion_label.setFont(QFont("Microsoft YaHei", 10))
        self.suggestion_label.setStyleSheet("""
            QLabel {
                color: #ffcc00;
                background-color: rgba(50, 50, 50, 0.9);
                padding: 8px;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.suggestion_label.setMinimumHeight(80)
        main_layout.addWidget(self.suggestion_label)

        # Reasoning
        self.reasoning_label = QLabel("")
        self.reasoning_label.setFont(QFont("Microsoft YaHei", 9))
        self.reasoning_label.setStyleSheet("color: #aaaaaa;")
        self.reasoning_label.setWordWrap(True)
        main_layout.addWidget(self.reasoning_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.play_btn = QPushButton("一键出牌")
        self.play_btn.clicked.connect(self.on_play_clicked)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #00C851;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #00e05c;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)
        self.play_btn.setEnabled(False)

        self.start_btn = QPushButton("开始识别")
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00C851;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #00e05c;
            }
        """)

        self.pass_btn = QPushButton("不出")
        self.pass_btn.clicked.connect(self.on_pass_clicked)
        self.pass_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffbb33;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #ffcc55;
            }
        """)

        settings_btn = QPushButton("设置")
        settings_btn.clicked.connect(self.on_settings_clicked)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #33b5e5;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #55c5eb;
            }
        """)

        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.pass_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(settings_btn)
        main_layout.addLayout(button_layout)

        self.setStyleSheet("""
            AIFloatingWindow {
                background-color: rgba(30, 30, 30, 0.9);
                border-radius: 10px;
                border: 1px solid #555555;
            }
        """)

        self.setLayout(main_layout)

    def nativeEvent(self, eventType, message):
        """Handle Windows WM_NCHITTEST for native title-bar dragging.

        Returns HTCAPTION for the title bar area so Windows handles
        dragging natively. No Qt mouse events needed — this eliminates
        interference with child widget click handling.
        """
        if sys.platform != 'win32':
            return False, 0

        try:
            msg = ctypes.cast(
                ctypes.c_void_p(int(message)),
                ctypes.POINTER(_MSG)
            ).contents

            if msg.message == WM_NCHITTEST:
                x, y = msg.pt.x, msg.pt.y
                local = self.mapFromGlobal(QPoint(x, y))
                # Title bar region: top ~42px, exclude close button area (~right 40px)
                if (10 <= local.x() <= self.width() - 40 and
                        10 <= local.y() <= 42):
                    return True, HTCAPTION
        except Exception:
            pass

        return False, 0

    def update_state(self, state: Dict):
        """Update UI with current game state."""
        if state.get('window_missing'):
            self.status_label.setText("⚠️ 请打开斗地主窗口")
            self.status_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 11px;")
            self.my_cards_label.setText("我的牌: 等待中...")
            self.my_cards_label.setStyleSheet("color: #ffcc00;")
            self.suggestion_label.setText("窗口未检测到\n\n请确保斗地主游戏窗口处于可见状态")
            self.suggestion_label.setStyleSheet("""
                QLabel {
                    color: #ff4444;
                    background-color: rgba(50, 50, 50, 0.9);
                    padding: 8px;
                    border-radius: 6px;
                    font-weight: bold;
                }
            """)
            self.opponent_label.setText("上家: -张  |  下家: -张")
            self.opponent_played_label.setText("上家出牌: -  |  下家出牌: -")
            self.play_btn.setEnabled(False)
            return

        my_role = state.get('my_role', 'unknown')
        my_count = state.get('my_count', 0)
        upper_count = state.get('upper_player_count', 0)
        lower_count = state.get('lower_player_count', 0)
        my_cards = state.get('my_cards', [])
        upper_last = state.get('upper_last', [])
        lower_last = state.get('lower_last', [])

        role_name = {'landlord': '地主', 'farmer': '农民'}.get(my_role, '未知')
        self.status_label.setText(f"当前身份: {role_name}  |  剩余: {my_count}张")
        self.status_label.setStyleSheet("color: #cccccc;")

        if my_cards:
            short_display = ' '.join(self._format_card(c) for c in sorted(
                my_cards, key=lambda x: self._get_rank_value(x)))
            if len(short_display) > 30:
                short_display = short_display[:27] + '...'
            self.my_cards_label.setText(f"我的牌: {short_display}")
            self.my_cards_label.setStyleSheet("color: #ffffff;")
            if self.suggestion_label.text().startswith("等待中"):
                self.suggestion_label.setText("识别中...")
                self.suggestion_label.setStyleSheet("""
                    QLabel {
                        color: #88ff88;
                        background-color: rgba(50, 50, 50, 0.8);
                        padding: 8px;
                        border-radius: 6px;
                    }
                """)
        else:
            self.my_cards_label.setText("我的牌: 等待中...")
            self.my_cards_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
            self.suggestion_label.setText("等待中...\n\n未检测到手牌，请确保游戏窗口可见")
            self.suggestion_label.setStyleSheet("""
                QLabel {
                    color: #ffcc00;
                    background-color: rgba(50, 50, 50, 0.9);
                    padding: 8px;
                    border-radius: 6px;
                    font-weight: bold;
                }
            """)
            self.reasoning_label.setText("")

        self.opponent_label.setText(f"上家: {upper_count}张  |  下家: {lower_count}张")

        if upper_last or lower_last:
            upper_str = ' '.join(self._format_card(c) for c in upper_last) if upper_last else '-'
            lower_str = ' '.join(self._format_card(c) for c in lower_last) if lower_last else '-'
            self.opponent_played_label.setText(f"上家出牌: {upper_str}  |  下家出牌: {lower_str}")
            self.opponent_played_label.setStyleSheet("color: #ff6666; font-weight: bold;")
        else:
            self.opponent_played_label.setText(f"上家出牌: -  |  下家出牌: -")
            self.opponent_played_label.setStyleSheet("color: #aaaaaa;")

        if state.get('current_turn') == 'self' and my_cards:
            self.play_btn.setEnabled(True)
        else:
            self.play_btn.setEnabled(False)

    def update_suggestion(self, decision: Dict):
        """Update UI with AI suggestion."""
        action = decision.get('action', 'pass')
        cards = decision.get('cards', [])
        play_type = decision.get('type', 'pass')
        confidence = decision.get('confidence', 0)
        reasoning = decision.get('reasoning', '')

        self.suggestion_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: rgba(50, 50, 50, 0.8);
                padding: 8px;
                border-radius: 6px;
            }
        """)

        if action == 'thinking':
            self.suggestion_label.setText(f"正在思考...\n\n{reasoning}")
            self.suggestion_label.setStyleSheet("""
                QLabel {
                    color: #88aaff;
                    background-color: rgba(50, 50, 50, 0.8);
                    padding: 8px;
                    border-radius: 6px;
                }
            """)
            self.reasoning_label.setText("")
        elif action == 'pass':
            self.suggestion_label.setText("建议: 不出 (pass)")
            self.reasoning_label.setText(reasoning)
        else:
            card_str = ' '.join(self._format_card(c) for c in cards)
            type_names = {
                'single': '单牌', 'pair': '对子', 'triple': '三张',
                'triple_with_single': '三带一', 'triple_with_pair': '三带一对',
                'bomb': '炸弹', 'rocket': '王炸', 'straight': '顺子',
                'consecutive_pairs': '连对'
            }
            type_cn = type_names.get(play_type, play_type)
            conf_pct = int(confidence * 100)
            self.suggestion_label.setText(f"出: {card_str}\n类型: {type_cn}  (置信度: {conf_pct}%)")
            self.reasoning_label.setText(reasoning)

    def _on_landlord_clicked(self, player: str):
        """Handle landlord button click."""
        # Update button styles
        for p, btn in self._landlord_buttons.items():
            if p == player:
                btn.setStyleSheet(self._landlord_btn_active_style)
            else:
                btn.setStyleSheet(self._landlord_btn_style)
        self._current_landlord = player
        self.landlord_changed.emit(player)

    def set_landlord(self, player: str):
        """Set landlord from external (e.g., from state manager)."""
        self._on_landlord_clicked(player)

    def set_opacity(self, opacity: float):
        """Set window opacity (0.0-1.0)."""
        self.setWindowOpacity(opacity)

    def on_close_clicked(self):
        self.hide()
        self.close_requested.emit()

    def on_play_clicked(self):
        self.play_clicked.emit()

    def on_pass_clicked(self):
        self.pass_clicked.emit()

    def on_settings_clicked(self):
        self.settings_clicked.emit()

    def on_start_clicked(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("识别中...")
        self.start_clicked.emit()

    def _format_card(self, card: str) -> str:
        if card.startswith('Joker'):
            return '小王' if card == 'Joker_B' else '大王'
        return card

    def _get_rank_value(self, card: str) -> int:
        rank_map = {
            '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '1': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15, 'J': 16, 'J': 17
        }
        first_char = card[0]
        if first_char == '1':
            return 10
        if card.startswith('Joker'):
            return 17 if 'R' in card else 16
        return rank_map.get(first_char, 0)


def run_standalone():
    """Run the UI standalone for testing."""
    app = QApplication(sys.argv)
    window = AIFloatingWindow()
    window.show()
    window.move(100, 100)
    window.set_opacity(0.92)

    test_state = {
        'my_role': 'farmer',
        'my_count': 8,
        'upper_player_count': 5,
        'lower_player_count': 6,
        'my_cards': ['3♠', '5♥', '7♦', '2♣', 'A♠', 'K♥', 'Joker_B', 'Joker_R'],
        'current_turn': 'self'
    }
    window.update_state(test_state)

    test_decision = {
        'action': 'play',
        'cards': ['Joker_B', 'Joker_R'],
        'type': 'rocket',
        'confidence': 0.95,
        'reasoning': 'Opponent has few cards left, use rocket to win immediately.'
    }
    window.update_suggestion(test_decision)

    sys.exit(app.exec_())


if __name__ == "__main__":
    run_standalone()
