"""
Doudizhu AI Assistant - Main Entry Point
Integrates all modules and runs the main capture/processing loop.
"""

import sys
import time
import yaml
import threading
from queue import Queue, Empty
from typing import Optional, Dict
import os

# Setup logging to file
LOG_FILE = open('doudizhu_log.txt', 'w', encoding='utf-8')

def log(msg):
    """Log to file and stdout."""
    LOG_FILE.write(msg + '\n')
    LOG_FILE.flush()
    print(msg, flush=True)

# Import torch first to avoid DLL loading issues
import torch

from PyQt5.QtWidgets import QApplication

from modules import (
    WindowCapture,
    CardRecognizer,
    GameStateManager,
    AIFloatingWindow
)
from douzero_ai import create_ai_engine


class DoudizhuAssistant:
    """Main controller for the Doudizhu AI Assistant."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.window_capture = WindowCapture()
        # Card recognition (YOLOv8 model)
        rec_conf = self.config.get('recognition', {})
        self.card_recognizer = CardRecognizer(
            model_path=rec_conf.get('model_path', 'models/yolov8_cards.pt'),
            confidence_threshold=rec_conf.get('confidence_threshold', 0.5)
        )
        self.state_manager = GameStateManager()
        self.ai_engine = create_ai_engine(
            strategy_mode=self.config.get('ai', {}).get('strategy', 'balanced')
        )
        self.ui: Optional[AIFloatingWindow] = None
        self.running = False
        self.started = False  # Will be set True when user clicks "开始识别"
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.frame_queue = Queue(maxsize=2)  # Bounded queue, drop old frames
        self.result_queue = Queue(maxsize=1)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            log(f"Warning: Could not load config: {e}, using defaults")
            return {
                'recognition': {
                    'template_dir': 'templates',
                    'confidence_threshold': 0.7,
                    'fps': 10,
                },
                'strategy_mode': 'balanced',
                'window_opacity': 0.92
            }

    def start(self) -> bool:
        """Start the assistant. Returns True if successful."""
        # Find game window
        hwnd = self.window_capture.find_window()
        if hwnd is None:
            log("ERROR: Could not find Doudizhu game window")
            log("Please open a Doudizhu game first")
            return False

        log(f"Found game window: {hwnd}")

        self.running = True
        self.state_manager.reset()
        self.card_recognizer.reset_history()  # Clear temporal smoothing history

        # Start background threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        log("Capture and processing threads started")
        return True

    def stop(self) -> None:
        """Stop the assistant."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """Background thread for capturing screenshots."""
        import time as time_module
        rec_config = self.config.get('recognition', {})
        target_fps = rec_config.get('fps', 10)
        frame_interval = 1.0 / target_fps
        last_debug_time = 0
        window_missing_since = None  # Track when window became missing

        while self.running:
            # Wait for user to click "开始识别"
            if not self.started:
                time_module.sleep(0.1)
                continue

            start_time = time_module.time()

            # Check if window still exists
            rect = self.window_capture.get_window_rect()
            if self.window_capture.hwnd is None or rect is None:
                if window_missing_since is None:
                    window_missing_since = time_module.time()
                    log(f"[Capture] Window not found, re-finding...")
                # Notify UI after 3 seconds of missing window
                elif time_module.time() - window_missing_since > 3:
                    # Put a special notification in result queue
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                    self.result_queue.put({
                        'state': {'window_missing': True},
                        'decision': None
                    })
                self.window_capture.find_window()
                time_module.sleep(0.5)
                continue

            window_missing_since = None  # Reset when window found

            # Capture frame
            screenshot = self.window_capture.capture()
            if screenshot is not None:
                # Drop old frame if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(screenshot)

                # Debug log every 5 seconds
                if time_module.time() - last_debug_time > 5:
                    last_debug_time = time_module.time()
                    log(f"[Capture] OK frame={screenshot.shape}, qsize={self.frame_queue.qsize()}, rect={rect}")
            else:
                log(f"[Capture] Capture returned None, rect={rect}")

            # Sleep to maintain FPS
            elapsed = time_module.time() - start_time
            if elapsed < frame_interval:
                time_module.sleep(frame_interval - elapsed)

    def _process_loop(self) -> None:
        """Background thread for processing frames (recognition + AI)."""
        import time
        last_log_time = 0
        last_cards_hash = None
        last_process_time = 0
        min_process_interval = 0.4  # max ~2.5 fps to prevent queue buildup

        while self.running:
            # Wait for user to click "开始识别"
            if not self.started:
                import time
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_process_time < min_process_interval:
                # Rate limit: skip frames arriving too fast
                try:
                    self.frame_queue.get(timeout=0.05)  # discard
                except Empty:
                    pass
                continue
            last_process_time = now

            try:
                screenshot = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                _t_start = time.perf_counter()

                # Get relative regions
                rect = self.window_capture.get_window_rect()
                if not rect:
                    continue

                regions = self.window_capture.get_relative_regions(rect)

                # Run recognition
                recognition_result = self.card_recognizer.recognize(screenshot, regions)
                _t_recog = time.perf_counter()

                # Debug opponent cards
                upper = recognition_result.get('upper_player_last', [])
                lower = recognition_result.get('lower_player_last', [])
                if upper or lower:
                    log(f"[Opponent] upper={upper}, lower={lower}")

                # Log every 5 seconds
                current_time = time.time()
                if current_time - last_log_time > 5:
                    last_log_time = current_time
                    card_count = len(recognition_result.get('my_cards', []))
                    cards_hash = hash(tuple(sorted(recognition_result.get('my_cards', []))))
                    log(f"[Process] Cards: {card_count}, hash={cards_hash}, frame_q={self.frame_queue.qsize()}, result_q={self.result_queue.qsize()}")

                # Update state manager
                self.state_manager.update_from_recognition(recognition_result)
                current_state = self.state_manager.get_state()

                # Get AI decision if we have cards
                decision = None
                current_cards = current_state.get('my_cards', [])
                if current_cards:
                    decision = self.ai_engine.decide(current_state)
                    log(f"[AI] cards={len(current_cards)}, decision={decision.get('action') if decision else None}, type={decision.get('type') if decision else None}")

                # Log timing every frame
                _t_end = time.perf_counter()
                recog_ms = (_t_recog - _t_start) * 1000
                total_ms = (_t_end - _t_start) * 1000
                ai_ms = (_t_end - _t_recog) * 1000
                log(f"[Perf] recog={recog_ms:.0f}ms ai={ai_ms:.0f}ms total={total_ms:.0f}ms")

                # Update UI from main thread needs to be handled via Qt signals
                result = {
                    'state': current_state,
                    'decision': decision
                }

                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put(result)

            except Exception as e:
                log(f"Error processing frame: {e}")
                continue

    def _run_ui(self) -> None:
        """Run the UI in the main thread."""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        self.ui = AIFloatingWindow()
        opacity = self.config.get('window_opacity', 0.92)
        self.ui.set_opacity(opacity)

        # Connect signals
        self.ui.close_requested.connect(self._on_close)
        self.ui.play_clicked.connect(self._on_play_clicked)
        self.ui.pass_clicked.connect(self._on_pass_clicked)
        self.ui.settings_clicked.connect(self._on_settings_clicked)
        self.ui.start_clicked.connect(self._on_start)
        self.ui.landlord_changed.connect(self._on_landlord_changed)

        # Position window
        screen_geometry = app.primaryScreen().availableGeometry()
        self.ui.move(screen_geometry.right() - self.ui.width() - 20,
                    screen_geometry.top() + 20)
        self.ui.show()

        # Timer to update UI from processing results
        from PyQt5.QtCore import QTimer

        _last_shown_decision = None  # Track last decision to avoid redundant updates

        def update_ui():
            nonlocal _last_shown_decision
            try:
                result = self.result_queue.get_nowait()
                state = result['state']
                decision = result['decision']
                if self.ui:
                    self.ui.update_state(state)
                    # Only update suggestion when decision actually changed
                    decision_key = (decision.get('action') if decision else None,
                                    tuple(sorted(decision.get('cards', [])) if decision else []),
                                    decision.get('type') if decision else None)
                    if decision_key != _last_shown_decision:
                        _last_shown_decision = decision_key
                        if decision:
                            self.ui.update_suggestion(decision)
                        elif state.get('my_cards'):
                            self.ui.update_suggestion({
                                'action': 'thinking',
                                'cards': [],
                                'type': 'processing',
                                'confidence': 0,
                                'reasoning': '正在分析最佳出牌方案...'
                            })
                        # Force widget repaint to ensure UI updates
                        self.ui.update()
                        log(f"[UI] Updated: cards={len(state.get('my_cards', []))}, decision={decision.get('action') if decision else None}")
            except Empty:
                pass
            except Exception as e:
                log(f"[UI] Error: {e}")

        timer = QTimer()
        timer.timeout.connect(update_ui)
        timer.start(100)  # Check for updates every 100ms

        app.exec_()

    def _on_close(self):
        """Handle window close."""
        self.stop()
        QApplication.quit()

    def _on_play_clicked(self):
        """Handle play button click."""
        # For future implementation: auto-click
        pass

    def _on_pass_clicked(self):
        """Handle pass button click."""
        # For future implementation: auto-click
        pass

    def _on_start(self):
        """Handle start button click - begin recognition."""
        log("[Start] _on_start called, setting started=True")
        self.started = True
        self.state_manager.reset()
        self.card_recognizer.reset_history()
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        log("[Start] Recognition started - waiting for new round")

    def _on_landlord_changed(self, player: str):
        """Handle landlord selection change."""
        log(f"[Landlord] Setting landlord to: {player}")
        self.state_manager.set_landlord(player)
        # After setting landlord, re-run AI on current state to update suggestion
        current_state = self.state_manager.get_state()
        my_cards = current_state.get('my_cards', [])
        if my_cards:
            decision = self.ai_engine.decide(current_state)
            try:
                self.result_queue.get_nowait()
            except Empty:
                pass
            self.result_queue.put({
                'state': current_state,
                'decision': decision
            })
            log(f"[Landlord] AI re-evaluated with role={current_state.get('my_role')}")

    def _on_settings_clicked(self):
        """Handle settings button click."""
        # TODO: Open settings dialog
        pass

    def run(self) -> None:
        """Run the full application."""
        if not self.start():
            sys.exit(1)

        self._run_ui()
        self.stop()


def main():
    """Main entry point."""
    log("=" * 50)
    log(" 斗地主AI Assistant ")
    log("=" * 50)
    log("\nStarting...")

    assistant = DoudizhuAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
